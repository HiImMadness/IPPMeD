import numpy as np
import pandas as pd
import cv2
import os
import glob
import torch

import nibabel as nib

def rle2mask(mask_rle: str, shape, label=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if mask_rle == 0:
        return np.zeros(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1] * shape[2], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def custom_dice(y_true, y_pred, label=1):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    label_positions = y_true_flat == label

    true_positives = np.sum(label_positions * y_pred_flat)
    false_positives = np.sum((1 - label_positions) * y_pred_flat)
    false_negatives = np.sum(label_positions * (1 - y_pred_flat))

    epsilon = 1e-7
    dice = (2.0 * true_positives + epsilon) / (
        2.0 * true_positives + false_positives + false_negatives + epsilon
    )

    return dice

def find_largest_containing_circle(segmentation, pixdim):
    pixdim = pixdim.cpu().numpy()
    largest_circle = None
    largest_slice = -1
    max_radius = -1
    segmentation8 = segmentation.astype(np.float32).astype('uint8')
    for i in range(segmentation8.shape[-1]):
        # Find the contours in the segmentation
        contours, _ = cv2.findContours(image = segmentation8[:,:,i], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Fit the smallest circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > max_radius:
                max_radius = radius
                largest_circle = ((int(x), int(y)), int(radius))
                largest_slice = i
    recist = max_radius * 2 * pixdim[0]
    #     print(max_radius)
    predicted_volume = np.round(np.sum(segmentation.flatten())*pixdim[0]*pixdim[1]*pixdim[2]*0.001,2)
    return recist, predicted_volume, largest_circle, largest_slice

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if np.array_equal(img, np.zeros(img.shape)):
      return 0

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def submission_gen(label, pixdim, name):
    """Create a submission csv from a path of segmentation prediction.
    predpath: Path of your fodler containing the predictions
    /!\ : The path should directly contain the .nii.gz files
    outputpath: Path of where the csv will be saved out"""
    #pred_files = glob.glob(f"{predpath}/*")
    label = label[0]
    shape_list = label.shape
    rle_list = mask2rle(label)
    (
        recist,
        predicted_volume,
        largest_circle,
        largest_slice,
    ) = find_largest_containing_circle(label, pixdim)
    if recist < 0:
        recist = 0
    recist_list = recist
    volume_list = predicted_volume
    filename = name .split("/")[-1]
    patient_id_list = filename.split(".")[0]
        
        
    df = {
            "id": patient_id_list[:-4],
            "rle": rle_list,
            "recist": recist_list,
            "volume": volume_list,
            "data_shape": shape_list,
        }
    return df

def score(
    submission: pd.DataFrame
) -> float:
    """
    ======================================================================================================================#

    inputs:
    For seg -> compute DICE
    for recist -> compute MAE
    for volume -> compute MAE

    Return Mean of the three metrics

    Doctest:
    #>>> import numpy as np
    #>>> import pandas as pd
    #>>> rle_pred = ['1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1']
    #>>> recist_pred = [20,10,30,40]
    #>>> vol_pred = [100,200,300,400]
    #>>> shapes = ['(3, 3, 2)','(3, 3, 2)','(3, 3, 2)','(3, 3, 2)']
    #>>> rle_true = ['4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2']
    #>>> recist_true = [5,25,30,89]
    #>>> vol_true = [100,400,368, 472]
    #>>> sub = pd.DataFrame({'id':[1,2,3,4], 'rle': rle_pred, 'volume': vol_pred, 'recist':recist_pred, 'data_shape': shapes})
    #>>> sol = pd.DataFrame({'id':[1,2,3,4], 'rle': rle_true, 'volume': vol_true, 'recist':recist_true, 'data_shape': shapes})
    #>>> score(sub, sol, 'id')
    """

    # Initialize Dice computer
    #     dice = Dice(average='macro', num_classes = 2)
    # Iterate throught rows of dataframe
    solution = pd.read_csv("/home/ids/ext-6344/IPPMeD/val_labels.csv")
    merged_df = pd.merge(submission, solution, on="id", suffixes=('_sub', '_sol'))
    seg_error = []
    recist_error = []
    vol_error = []
    for index, row in merged_df.iterrows():
        # Convert rle to mask
        sub_array = rle2mask(
            row["rle_sub"], np.array(row["data_shape_sub"])
        )
        sol_array = rle2mask(
            row["rle_sol"], np.fromstring(row["data_shape_sol"][1:-1], sep=",", dtype="int")
        )
        sol_array = np.expand_dims(sol_array, axis=(0, 1))
        sub_array = np.expand_dims(sub_array, axis=(0, 1))
        # Compute Dice, recist and volume and store them
        seg_error = np.append(
            seg_error, custom_dice(y_true=sol_array, y_pred=sub_array)
        )
        recist_error = np.append(recist_error, np.abs(row["recist_sub"] - row["recist_sol"]) / row["recist_sol"])
        vol_error = np.append(vol_error, np.abs(row["volume_sub"] - row["volume_sol"]) / row["volume_sol"])
    # Rescale vol_error and recist_error to be the same order of magnitude by tresholding
    recist_error = np.where(recist_error > 1, 1, recist_error)
    vol_error = np.where(vol_error > 1, 1, vol_error)
    # Make sure error is maxed if array contains a repetition of the same element.
    score = np.mean([1 - np.mean(seg_error), np.mean(recist_error), np.mean(vol_error)])

    return score, 1 - np.mean(seg_error), np.mean(recist_error), np.mean(vol_error)


