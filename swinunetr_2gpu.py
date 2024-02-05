import os
import shutil
import tempfile
import wandb
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Invertd,
    Lambda,
    AsDiscreted
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
import pandas as pd 

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
    DistributedSampler,
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from evaluation_metric import submission_gen, score

torch.set_float32_matmul_precision('medium')

root_dir = os.getcwd()
print(root_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logger = WandbLogger(project="IPPMed", name="SwinUNETR")

class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 1000
        self.check_val = 20
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []
        self.losses = []

    def forward(self, x):
        return self._model(x)

    def setup(self, stage=None):
        # prepare data
        path = "/home/ssd/ext-6344"
        dir = os.path.join(path, 'data_challenge', 'train')
        image_dir = os.path.join(dir, 'volume')
        label_dir = os.path.join(dir, 'seg')
        image_paths = sorted([os.path.join(image_dir, filename) for filename in os.listdir(image_dir)])
        label_paths = sorted([os.path.join(label_dir, filename) for filename in os.listdir(label_dir)])
        assert len(image_paths) == len(label_paths)
        
        files = [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]
        
        train_files, val_files = train_test_split(files, test_size=0.05, random_state=42)
        print(len(train_files))
        
        
        

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"],image_only=False),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
            ]
        )
        
        def extract_second_label(data):
            # Assuming data["pred"] is of shape [batch_size, num_classes, ...]
            # and we need to extract the second label (index 1)
            data["second_label"] = data["pred"][1 , ...]
            return data
        
        self.post_proc = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=val_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True, to_onehot=2),
                Lambda(extract_second_label),
                #KeepLargestConnectedComponentd(
                #    keys="second_label",
                #    applied_labels=[1],  # Update the applied_labels depending on your use case
                #    independent=True,
                #    connectivity=None
                #),
            ]
        )

        
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_num=276,
            cache_rate=1.0,
            num_workers=8,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=15,
            cache_rate=1.0,
            num_workers=8,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        print("train ok")
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        print("test ok")
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.losses.append(loss.item())
        return {"loss": loss}

    def on_train_epoch_end(self):
        avg_loss = sum(self.losses)/len(self.losses)
        self.logger.experiment.log({'avg_train_loss': avg_loss,
                       'epoch': self.current_epoch
                       })
        self.epoch_loss_values.append(avg_loss)
        print(f"Average training loss at epoch {self.current_epoch}: {avg_loss}")
        self.losses = []


    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        metadata = batch['image_meta_dict']
        pixdim = metadata['pixdim'][:, [1, 2, 3]][0]
        name = metadata['filename_or_obj'][0]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        batch["pred"] = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(batch["pred"], labels)
        label =  [self.post_proc(i)["second_label"] for i in decollate_batch(batch)]
        outputs = [self.post_pred(i) for i in decollate_batch(batch["pred"])]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        df = submission_gen(label, pixdim, name)
        
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        d.update(df)
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        rle_list = []
        recist_list = []
        volume_list = []
        patient_id_list = []
        shape_list = []
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
            rle_list.append(output["rle"])
            patient_id_list.append(output["id"])
            recist_list.append(output["recist"])
            volume_list.append(output["volume"])
            shape_list.append(output["data_shape"])
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        

        
        df = pd.DataFrame(
            {
                "id": patient_id_list,
                "rle": rle_list,
                "recist": recist_list,
                "volume": volume_list,
                "data_shape": shape_list,
            }
        )
        scor, seg_error, recist_error, vol_error = score(df)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
            "epoch": self.current_epoch,
            "score": scor,
            "seg_error": seg_error,
            "recist_error": recist_error,
            "vol_error": vol_error
        }
        self.logger.experiment.log(tensorboard_logs)
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory


net = Net()

# set up checkpoints
checkpoint_callback = ModelCheckpoint(dirpath=root_dir,
                                      filename="SwinUNETR-{epoch}",
                                      verbose=True,
                                      every_n_epochs=5,
                                      save_top_k=-1
)

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    precision = '16-mixed',
    devices=[0,1],
    strategy='ddp',
    max_epochs=net.max_epochs,
    check_val_every_n_epoch=net.check_val,
    callbacks=[checkpoint_callback],
    default_root_dir=root_dir,
    logger = logger
)

# train
trainer.fit(net)