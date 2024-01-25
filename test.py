import matplotlib.pyplot as plt
from swinunetr_1gpu import Net  # Make sure this import is correct
import os
import torch
from monai.inferers import sliding_window_inference  # Import sliding_window_inference
from evaluation_metric import score
import pandas as pd

root_dir = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(score(pd.read_csv("/home/ids/ext-6344/IPPMeD/val_labels.csv")))


        