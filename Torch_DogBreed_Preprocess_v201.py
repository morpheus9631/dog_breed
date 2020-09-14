# Kaggle: https://www.kaggle.com/c/dog-breed-identification
# Author: Morpheus Hsieh

from __future__ import print_function, division

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

def getCsvLabels(path):
    fname = 'labels.csv'
    f_abspath = join(path, fname)
    if not exists(f_abspath): 
        raise "'{}' not exist...".format(f_abspath)
    df = pd.read_csv(f_abspath)
    return df

def countNumEachBreeds(df):
    df1 = df.groupby('breed').count().sort_values(by='id', ascending=False)
    df1.insert(0, 'breed', df1.index)
    df1 = df1.rename(columns={'id': 'count'})
    df1 = df1.reset_index(drop=True)
    df1.insert(0, 'breed_id', df1.index)
    return df1

def writeToCsv_BreedIds(path, df):
    fname = 'breeds_processed.csv'
    f_abspath = join(path, fname)
    isExist = exists(f_abspath)
    if isExist:
        msg = "'{}' already exist, skipped...".format(f_abspath)
    else:
        df.to_csv(f_abspath, index=False)
        msg = "'{}' saved.".format(f_abspath)
    return exists(f_abspath), msg

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

def writeToCsv_ProcessedLabels(path, df):
    fname = 'labels_processed.csv'
    f_abspath = join(path, fname)
    isExist = exists(f_abspath)
    if isExist:
        msg = "'{}' already exist, skipped...".format(f_abspath)
    else:
        df.to_csv(f_abspath, index=False)
        msg = "'{}' saved.".format(f_abspath)
    return exists(f_abspath), msg


def writeToNpzFile(path, df, types='breeds'):
    if types == 'breeds':
        fname = 'breeds_processed.npz'
    elif types == 'labels':
        fname = 'labels_processed.npz'
    else:
        raise "Types must be 'breeds' or 'labels'..."

    f_abspath = join(path, fname)
    isExist = exists(f_abspath)
    if isExist:
        msg = "'{}' already exist, skipped...".format(f_abspath)
    else:
        col_names = df.columns.tolist()
        args = { x: df[x] for x in col_names }
        np.savez(f_abspath, **args)
        msg = "'{}' saved.".format(f_abspath)
    return exists(f_abspath), msg

def showNpzFile(f_abspath, num=10):
    load_data = np.load(f_abspath, allow_pickle=True)

    print('Images:')
    print('\n'.join('  '+load_data['images'][:num]))
    print('Labels:\n  {}'.format(load_data['labels'][:num]))
    return


def main():
    RawPath = r'D:\GitWork\dog_breed\data\raw'
    print('\nRaw path:', RawPath)

    TrainPath = join(RawPath, 'train')
    TestPath  = join(RawPath, 'test')
 
    ProcPath = r'D:\GitWork\dog_breed\data\processed'
    print('Proc path:', ProcPath)

    # Load labels.csv
    df_labels = getCsvLabels(RawPath)
    print('\nLabels.csv head:'); print(df_labels.info())
    print('\nLabels.csv info:'); print(df_labels.head()) 

    # Aggregate the number of each breeds
    df_breeds = countNumEachBreeds(df_labels)  # dataframe of breeds
    print('\nNum of each breeds:'); print(df_breeds)

    # Write breed ids and countr of each breeds to CSV
    isExist, msg = writeToCsv_BreedIds(ProcPath, df_breeds)
    print('\n',msg)

    # Write breed dictionary to npz
    isExist, msg = writeToNpzFile(ProcPath, df_breeds, 'breeds')
    print('\n',msg, isExist)

    # Convert labels to images path and breed id 
    img_path = join(RawPath, 'train')
    df_proc_lbls = createProcessedLabels(img_path, df_labels, df_breeds)
    print('\nProcessed labels:'); print(df_proc_lbls.head())

    # Write processed labels to csv file
    isExist, msg = writeToCsv_ProcessedLabels(ProcPath, df_proc_lbls)
    print('\n',msg)

    # Write labels to npz
    isExist, msg = writeToNpzFile(ProcPath, df_proc_lbls, 'labels')
    print('\n',msg, isExist)


if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
