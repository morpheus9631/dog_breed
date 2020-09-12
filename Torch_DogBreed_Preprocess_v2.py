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

def writeToCsv_BreedCounts(path, df):
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

def createDataList(df, frac_for_train=0.8, disp=False):
    total_rows = df.shape[0]
    train_len = int(float(frac_for_train) * float(total_rows))
    # print('Train len: ', train_len)

    train_imgs = df['image'][:train_len]
    train_lbls = df['breed_id'][:train_len]

    valid_imgs = df['image'][train_len:]
    valid_lbls = df['breed_id'][train_len:]

    if disp:
        print('\nTotal rows:', total_rows)
        formatter = '{} size: images ({}), lables ({})'
        print(formatter.format('train', len(train_imgs), len(train_lbls)))
        print(formatter.format('valid', len(valid_imgs), len(valid_imgs)))

    return {
        'train': { 'images': train_imgs, 'labels': train_lbls },
        'valid': { 'images': valid_imgs, 'labels': valid_lbls }
    }

def saveToNpzFile(path, train_imgs, train_lbls, valid_imgs, valid_lbls):
    phase = ['train', 'valid']
    types = ['images', 'labels']

    fnames = ['{}_data.npz'.format(x) for x in phase]
    data = [[train_imgs, train_lbls], [valid_imgs, valid_lbls]]

    print('Process start...')
    result = {}
    for i in range(len(fnames)):
        f_abspath = join(path, fnames[i])
        print("'{}' processing...".format(f_abspath))
        args = { types[0]: data[i][0], types[1]: data[i][1] }
        np.savez(f_abspath, **args)
        result[f_abspath] = exists(f_abspath)
    print('Process end.')
    return result


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

    # Write number of each breeds to CSV
    isExist, msg = writeToCsv_BreedCounts(ProcPath, df_breeds)
    print('\n',msg)

    # Convert labels to images path and breed id 
    img_path = join(RawPath, 'train')
    df_proc_lbls = createProcessedLabels(img_path, df_labels, df_breeds)
    print('\nProcessed labels:'); print(df_proc_lbls.head())

    # Write processed labels to csv file
    isExist, msg = writeToCsv_ProcessedLabels(ProcPath, df_proc_lbls)
    print('\n',msg)
    
    # Build data lists
    FRAC_FOR_TRAIN = 0.8
    data = createDataList(df_proc_lbls, FRAC_FOR_TRAIN, True)
    train_imgs = data['train']['images']
    train_lbls = data['train']['labels']
    valid_imgs = data['valid']['images'] 
    valid_lbls = data['valid']['labels']

    # Verify npz file
    print('\nWrite to npz files:')
    results = saveToNpzFile(ProcPath, train_imgs, train_lbls, valid_imgs, valid_lbls)

    print('\nNPZ files:')
    for f_abspath in results:
        outstr = 'exist.' if exists(f_abspath) else 'not exist...'
        print("\n'{}' {}".format(f_abspath, outstr))
        showNpzFile(f_abspath)

if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
