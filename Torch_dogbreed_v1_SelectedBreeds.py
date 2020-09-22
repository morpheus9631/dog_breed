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



def main():
    DataPath = r'D:\GitWork\dog_breed\data'
    print('\nRaw path:', DataPath)
 
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

    #
    # Create processed labels file
    img_path = join(DataPath, 'train')
    df_labels_proc = createProcessedLabels(img_path, df_labels, df_breeds_proc)
    print('\nProcessed labels:'); 
    print(df_labels_proc.info())
    print(df_labels_proc.head())

    #
    # Get most popular breeds
    NumClasses = 16
    df1 = df_breeds_proc.sort_values(['count', 'breed'], ascending=(False, True))
    df1 = df1.head(NumClasses)
    # print('\nSelected breeds:')
    # print(df1.info()); print(df1.head())

    selected_breed_list = list(df1['breed'])
    formatter = '\nSelected breeds: [\n  {}\n]'
    print(formatter.format('\n  '.join(selected_breed_list)))

    selected_bid_list = list(df1['breed_id'])
    formatter = '\nSelected breed ids:\n  {}'
    print(formatter.format(selected_bid_list))

    #
    # Get selected labels according to selected breed ids
    df2 = df_breeds_proc.copy()
    df2_selected = df2[df2['breed'].isin(selected_breed_list)]
    print('\nSelected breeds processed data:') 
    print(df2_selected.info())
    print(df2_selected.head())

    df3 = df_labels_proc.copy()
    df3_selected = df3[df3['breed_id'].isin(selected_bid_list)]
    print('\nSelected labels processed data:')
    print(df3_selected.info())
    print(df3_selected.head())


if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
