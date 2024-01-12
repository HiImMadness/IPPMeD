import matplotlib.pyplot as plt
from swinunetr_1gpu import Net  # Make sure this import is correct
import os
import torch
from monai.inferers import sliding_window_inference  # Import sliding_window_inference

root_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Net.load_from_checkpoint(os.path.join(root_dir, "SwinUNETR-medium-epoch=839.ckpt"))
net.prepare_data()
net.eval()
net.to(device)  # Use the device directly

slice_map = 100

for case_num in range(len(net.val_ds)):
    with torch.no_grad():
        img_name = os.path.split(net.val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = net.val_ds[case_num]["image"]
        label = net.val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).to(device)
        val_labels = torch.unsqueeze(label, 1).to(device)
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, net, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image: {img_name}")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map])
        plt.savefig(os.path.join(root_dir, f"fig{case_num}.png"))