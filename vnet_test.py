import os
import shutil
import tempfile
import wandb
from pytorch_lightning.loggers import WandbLogger

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
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
    KeepLargestConnectedComponentd,
    KeepLargestConnectedComponent,
    SaveImaged,
    SaveImage
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import VNet
from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
    DistributedSampler,
    Dataset
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from monai.data.utils import decollate_batch

torch.set_float32_matmul_precision('medium')

root_dir = os.getcwd()
print(root_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        '''
        self._model = VNet(
            spatial_dims=3,  # 3D model
            in_channels=1,   # Number of input channels
            out_channels=2,  # Number of output channels
        ).to(device)
        '''
        self._model = UNet(
            spatial_dims=3,  # 3D UNet
            in_channels=1,   # Number of input channels
            out_channels=2,  # Number of output channels
            channels=(16, 32, 64, 128, 256), # Number of channels per layer
            strides=(2, 2, 2, 2),            # Stride for each layer
            num_res_units=2, # Number of residual units
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 1300
        self.check_val = 10
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []
        self.losses = []

    def forward(self, x):
        return self._model(x)
    
    def prepare_data(self):
        # prepare data
        path = "/home/ssd/ext-6344"
        test_dir = os.path.join(path, 'data_challenge', 'train')
        
        test_image_dir = os.path.join(test_dir, 'volume')
        
        test_files = [{"image": test_image_dir + "/" + img} for img in sorted(os.listdir(test_image_dir))]


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
            ]
        )
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear"),
                ),
                # Add other necessary transforms for test data
            ]
        )

        self.test_ds = Dataset(
            data=test_files,
            transform=test_transforms,
        )
        def extract_second_label(data):
            # Assuming data["pred"] is of shape [batch_size, num_classes, ...]
            # and we need to extract the second label (index 1)
            data["second_label"] = data["pred"][1 , ...]
            return data
        post_transforms = Compose(
            [
                Invertd(
                    keys="pred",
                    transform=test_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True),
                KeepLargestConnectedComponentd(
                    keys="pred",
                    is_onehot=False,
                    applied_labels=[1],  # Update the applied_labels depending on your use case
                    independent=True,
                    connectivity=3,
                    num_components=1
                ),
                SaveImaged(keys="pred", 
                           meta_keys="pred_meta_dict",
                           output_dir="/home/ssd/ext-6344/IPPMeD/test",
                           output_postfix="seg",
                           resample=False,
                           separate_folder=False
                           ),
            ]
        )
        return post_transforms

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        self.losses.append(loss.item())
        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_end(self):
        avg_loss = sum(self.losses)/len(self.losses)
        wandb.log({'avg_train_loss': avg_loss,"epoch":self.current_epoch})
        self.epoch_loss_values.append(avg_loss)
        print(f"Average training loss at epoch {self.current_epoch}: {avg_loss}")
        self.losses = []

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
            "epoch":self.current_epoch
        }
        wandb.log(tensorboard_logs)
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
        return {"log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        images = batch["image"].to(device)
        roi_size = (96, 96, 96)
        sw_batch_size = 4

        # Perform inference using sliding window
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        # Decollate the batched outputs
        decollated_outputs = decollate_batch(outputs)

        # Apply AsDiscrete to each output in the decollated list
        discrete_outputs = [AsDiscrete(to_onehot=1)(output) for output in decollated_outputs]

        return discrete_outputs


    
def save_nifti(outputs, output_dir):
    saver = SaveImaged(keys='pred', output_dir=output_dir, output_postfix='pred', resample=False)
    for output in outputs:
        saver({'pred': output})

def run_test(net, device):
    path = "/home/ssd/ext-6344/IPPMeD/test"
    test_loader = DataLoader(net.test_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

    net.eval()
    net.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader)):
            outputs = net.test_step(batch, batch_idx)
            save_nifti(outputs, path)


net = Net.load_from_checkpoint(os.path.join(root_dir, "UNET-epoch=3379.ckpt"))
post_transforms = net.prepare_data()
test_loader = DataLoader(net.test_ds, batch_size=1, num_workers=4)

def extract_second_label(data):
    # Assuming data["pred"] is of shape [batch_size, num_classes, ...]
    # and we need to extract the second label (index 1)
    data["second_label"] = data["pred"][1 , ...]
    print("After extract_second_label:", data["second_label"].shape)
    return data

with torch.no_grad():
    for test_data in test_loader:
        test_inputs = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, net)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]