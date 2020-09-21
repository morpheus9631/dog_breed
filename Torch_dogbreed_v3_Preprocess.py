# Author: Morpheus Hsieh

from __future__ import print_function, division

import os, sys
import errno
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pandas import Series, DataFrame
from os.path import join, exists

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms, utils


def getCsvLabels(path, fname):
    f_abspath = join(path, fname)
    try:
        df = pd.read_csv(f_abspath)
        return df
    except FileNotFoundError:
        print('\n', FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), f_abspath
        ))
        sys.exit()


def createProcessedBreeds(df):
    df1 = df.groupby('breed').count().sort_values(by='id', ascending=False)
    df1.insert(0, 'breed', df1.index)
    df1 = df1.rename(columns={'id': 'count'})
    df1 = df1.reset_index(drop=True)
    df1.insert(0, 'breed_id', df1.index)
    return df1


def createProcessedLabels(path, df_lbls, df_bds):
    df = pd.DataFrame(columns=['image', 'breed_id'])

    mapping = dict(df_bds[['breed', 'breed_id']].values)
    df['breed_id'] = df_lbls.breed.map(mapping)

    # Verify image exist or not accaording to id
    def id2ImgPath(path, ext='.jpg'):
        return (
            lambda f: join(path, f+ext) \
            if exists(join(path, f+ext)) else None
        )
    id2imgP = id2ImgPath(path)

    SersId = Series.to_numpy(df_lbls['id'])
    df['image'] = [id2imgP(v) for v in SersId]
    return df


def splitDataset(df, frac_for_train=0.8):
    total_rows = df.shape[0]
    train_len = int(float(frac_for_train) * float(total_rows))
    valid_len = total_rows - train_len

    df_train = df.head(train_len).copy()
    df_valid = df.tail(valid_len).copy()
    return { 'train': df_train, 'valid': df_valid }


def saveToNpzFile(f_abspath, df):
    col_names = df.columns.tolist()
    args = { x: df[x] for x in col_names }
    np.savez(f_abspath, **args)
    return exists(f_abspath)


def main():
    RawPath = r'D:\GitWork\dog_breed\data\raw'
    print('\nRaw path:', RawPath)
 
    ProcPath = r'D:\GitWork\dog_breed\data\processed'
    print('Proc path:', ProcPath)

    # Load labels.csv
    fname = 'labels.csv'
    df_labels = getCsvLabels(RawPath, fname)
    print("\n'{}':".format(fname))
    print('Info:'); print(df_labels.head()) 
    print('\nHead:'); print(df_labels.info())

    #
    # Create processed breeds dataframe 
    df_breeds_proc = createProcessedBreeds(df_labels)
    print('\nProcessed breeds:')
    print('Info:'); print(df_breeds_proc.info()) 
    print('\nHead:'); print(df_breeds_proc.head()) 

    #
    # Create processed labels file
    img_path = join(RawPath, 'train')
    df_labels_proc = createProcessedLabels(img_path, df_labels, df_breeds_proc)
    print('\nProcessed labels:'); 
    print(df_labels_proc.info())
    print(df_labels_proc.head())

    # Build data lists
    FRAC_FOR_TRAIN = 0.8
    df_data = splitDataset(df_labels_proc, FRAC_FOR_TRAIN)

    # Save data to .npz file
    print()
    phase = ['train', 'valid']
    for x in phase:
        df_data_save = df_data[x]
        fname = '{}_data.npz'.format(x)
        f_abspath = join(ProcPath, fname)
        saveToNpzFile(f_abspath, df_data_save)
        print("'{}' exist ? {}".format(f_abspath, exists(f_abspath)))


if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
