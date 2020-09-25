# Kaggle: https://www.kaggle.com/c/dog-breed-identification
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
from os.path import join, exists, isfile

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


def writeToCsv(path, fname, df):
    f_abspath = join(path, fname)
    df.to_csv(f_abspath, index=False)
    return exists(f_abspath)


def createProcessedLabels(path, df_lbls, df_bds):
    df = pd.DataFrame(columns=['image', 'breed_id'])

    mapping = dict(df_bds[['breed', 'breed_id']].values)
    df['breed_id'] = df_lbls.breed.map(mapping)

    # Verify image exist or not accaording to id
    # I don't write full path in here, but id only. 
    # Because path can be as one parameter of customer dataset.
    def id2ImgPath(path, ext='.jpg'):
        return (
            lambda f: f \
            if exists(join(path, f+ext)) and isfile(join(path, f+ext)) else None
        )
    id2imgP = id2ImgPath(path)

    SersId = Series.to_numpy(df_lbls['id'])
    df['image'] = [id2imgP(v) for v in SersId]
    return df


def saveToNpz(path, fname, df):
    f_abspath = join(path, fname)
    col_names = df.columns.tolist()
    args = { x: df[x] for x in col_names }
    np.savez(f_abspath, **args)
    return exists(f_abspath)


def showNpzFile(f_abspath, num=10):
    load_data = np.load(f_abspath, allow_pickle=True)

    print('Images:')
    print('\n'.join('  '+load_data['images'][:num]))
    print('Labels:\n  {}'.format(load_data['labels'][:num]))
    return


def main():
    DataPath = r'D:\Dataset\dog-breed-identification'
    print('\nData path:', DataPath)
 
    ProcPath = r'D:\GitWork\dog_breed\processed'
    print('Proc path:', ProcPath)

    # Load labels.csv
    fname = 'labels.csv'
    df_labels = getCsvLabels(DataPath, fname)
    print("\n'{}':".format(fname))
    print('Info:'); print(df_labels.head()) 
    print('\nHead:'); print(df_labels.info())

    #
    # Create processed breeds dataframe 
    df_breeds_proc = createProcessedBreeds(df_labels)
    print('\nProcessed breeds:')
    print('Info:'); print(df_breeds_proc.info()) 
    print('\nHead:'); print(df_breeds_proc.head()) 

    # Write processed breeds data to CSV
    fname = 'breeds_processed.csv'
    isExist = writeToCsv(ProcPath, fname, df_breeds_proc)
    print("\n'{}' saved? {}".format(fname, isExist))

    # Save processed breeds data to npz 
    fname = 'breeds_processed.npz'
    isExist = saveToNpz(ProcPath, fname, df_breeds_proc)
    print("\n'{}' saved? {}".format(fname, isExist))

    #
    # Create processed labels file
    img_path = join(DataPath, 'train')
    df_labels_proc = createProcessedLabels(img_path, df_labels, df_breeds_proc)
    print('\nProcessed labels:'); 
    print(df_labels_proc.info())
    print(df_labels_proc.head())

    # Write processed labels to csv file
    fname = 'labels_processed.csv'
    isExist = writeToCsv(ProcPath, fname, df_labels_proc)
    print("\n'{}' saved? {}".format(fname, isExist))

    # Save processed labels data to npz 
    fname = 'labels_processed.npz'
    isExist = saveToNpz(ProcPath, fname, df_labels_proc)
    print("\n'{}' saved? {}".format(fname, isExist))



if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
