import matplotlib.pyplot as plt
from monai_test import Net
import os

root_dir = root_dir = os.getcwd()

net = Net()
net.load_from_checkpoint(root_dir+"/UNETR-medium-epoch=3079.ckpt")
net.eval()
net.to(device)

slice_map = 100
case_num = 4


with torch.no_grad():
    img_name = os.path.split(net.val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = net.val_ds[case_num]["image"]
    label = net.val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, net, overlap=0.8)
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
    plt.savefig(root_dir+"/fig.png")