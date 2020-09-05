# Topic: Dog Breed Identification
# From: https://www.kaggle.com/c/dog-breed-identification
# Author: Morpheus Hsieh (morpheus.hsieh@gmail.com)

from __future__ import print_function, division

import os, sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pandas import Series, DataFrame

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils

from configs.config_train import get_cfg_defaults


def id2ImgPath(path, ext='.jpg'):
    return (
        lambda f: os.path.join(path, f+ext) \
        if os.path.exists(os.path.join(path, f+ext)) else None
    )


def getBreedDict(series):
    # Collating the breed classes
    cls_set = set(series)
    cls_set_len = len(cls_set)
    print('Breed class: ', cls_set_len)

    # Create breed dict, both forward and backward dict
    cls_list = list(cls_set)
    cls_list.sort()
    breed_dict = { v:i for i, v in enumerate(cls_list) }
    return breed_dict

def getTrainValidData(img_list, bid_list, frac_for_train=0.8):
    total_rows = len(img_list)
    train_len = int(float(frac_for_train) * float(total_rows))
    print('\nTrain len: ', train_len)
    print('Valid len: ', (total_rows - train_len))

    train_imgs = img_list[:train_len]
    valid_imgs = img_list[train_len:]

    train_lbls = bid_list[:train_len]
    valid_lbls = bid_list[train_len:]

    return [train_imgs, valid_imgs, train_lbls, valid_lbls]


def main():
    RawPath = r'D:\GitWork\dog_breed\data\raw'
    print('Raw path: ', RawPath)

    label_fname = 'labels.csv'
    df = pd.read_csv(os.path.join(RawPath, label_fname))

    csv_columns = list(df.columns)
    print('\nColumns: ', csv_columns)

    print(); print(df.info())
    print(); print(df.head())

    # Verify image exist or not
    img_path = os.path.join(RawPath, 'train')
    id2imgP = id2ImgPath(img_path)

    SersId = Series.to_numpy(df["id"])
    img_list = [id2imgP(v) for v in SersId]
    df['image'] = img_list

    disp_num = 10
    # print('Top %d data of dataframe:'%disp_num)
    # print(df.head(disp_num))

    # if image not exist?
    cnt_no_img = sum(x is None for x in img_list)
    print('\nCount of none imgs: ', cnt_no_img)

    # Create breed dict
    SersBreed = Series.to_numpy(df["breed"]) 
    breed_dict = getBreedDict(SersBreed)

    print('\nBreed dict:')
    print(json.dumps(breed_dict, indent=4))

    # Append breed ID to dataframe
    bid_list = [breed_dict[b] for b in SersBreed]
    df['breed_id'] = bid_list

    print('\nTop %d data of dataframe:'%disp_num)
    print(df.head(disp_num))

    # save information to csv
    ProcPath = r'D:\GitWork\dog_breed\data\processed'

    csv_processed = os.path.join(ProcPath, 'processed_labels.csv')
    print("\nProcessed csv: '{}'".format(csv_processed))
    df.to_csv(csv_processed, index=False)

    # Split total rows to train and valid rows
    FracForTrain = 0.8
    print('Frac for train: ', FracForTrain)

    total_rows = df.shape[0]
    print('\nTotal rows: ', total_rows)

    npy_data = getTrainValidData(
        img_list, bid_list, FracForTrain
    )

    print('\nTop %d train images:'%disp_num)
    train_imgs = npy_data[0]
    print('\n'.join(train_imgs[:disp_num]))

    print('\nTop %d valid images:'%disp_num)
    valid_imgs = npy_data[1]
    print('\n'.join(valid_imgs[:disp_num]))

    print('\nTop %d train labels:'%disp_num)
    train_lbls = npy_data[2]
    print(train_lbls[:disp_num])

    print('\nTop %d valid labels:'%disp_num)
    valid_lbls = npy_data[3]
    print(valid_lbls[:disp_num])
    
    # Save numpy array as .npy file
    phase = ['train', 'valid']
    types = ['imgs', 'labels']
    fname = ['{}_{}.npy'.format(y, x) for x in types for y in phase]

    print('\nProcess start...')
    for i in range(len(fname)):
        f_abspath = os.path.join(ProcPath, fname[i])
        print("'{}' processing...".format(f_abspath))
        np.save(f_abspath, npy_data[i])
    print('Process end.')

    return


if __name__=='__main__':
    main()