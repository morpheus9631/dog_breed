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


def writeToCsv(path, fname, df):
    f_abspath = join(path, fname)
    df.to_csv(f_abspath, index=False)
    return exists(f_abspath)


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


def saveToNpz(path, fname, df):
    f_abspath = join(path, fname)
    col_names = df.columns.tolist()
    args = { x: df[x] for x in col_names }
    np.savez(f_abspath, **args)
    return exists(f_abspath)




# def createDataList(df, frac_for_train=0.8, disp=False):
#     total_rows = df.shape[0]
#     train_len = int(float(frac_for_train) * float(total_rows))
#     # print('Train len: ', train_len)

#     train_imgs = df['image'][:train_len]
#     train_lbls = df['breed_id'][:train_len]

#     valid_imgs = df['image'][train_len:]
#     valid_lbls = df['breed_id'][train_len:]

#     if disp:
#         print('\nTotal rows:', total_rows)
#         formatter = '{} size: images ({}), lables ({})'
#         print(formatter.format('train', len(train_imgs), len(train_lbls)))
#         print(formatter.format('valid', len(valid_imgs), len(valid_imgs)))

#     return {
#         'train': { 'images': train_imgs, 'labels': train_lbls },
#         'valid': { 'images': valid_imgs, 'labels': valid_lbls }
#     }


# def saveToNpzFile(path, train_imgs, train_lbls, valid_imgs, valid_lbls):
#     phase = ['train', 'valid']
#     types = ['images', 'labels']

#     fnames = ['{}_data.npz'.format(x) for x in phase]
#     data = [[train_imgs, train_lbls], [valid_imgs, valid_lbls]]

#     print('Process start...')
#     result = {}
#     for i in range(len(fnames)):
#         f_abspath = join(path, fnames[i])
#         print("'{}' processing...".format(f_abspath))
#         args = { types[0]: data[i][0], types[1]: data[i][1] }
#         np.savez(f_abspath, **args)
#         result[f_abspath] = exists(f_abspath)
#     print('Process end.')
#     return result


def showNpzFile(f_abspath, num=10):
    load_data = np.load(f_abspath, allow_pickle=True)

    print('Images:')
    print('\n'.join('  '+load_data['images'][:num]))
    print('Labels:\n  {}'.format(load_data['labels'][:num]))
    return


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
    #
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
    #
    img_path = join(RawPath, 'train')
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





    # # Build data lists
    # FRAC_FOR_TRAIN = 0.8
    # data = createDataList(df_proc_lbls, FRAC_FOR_TRAIN, True)
    # train_imgs = data['train']['images']
    # train_lbls = data['train']['labels']
    # valid_imgs = data['valid']['images'] 
    # valid_lbls = data['valid']['labels']

    # # Verify npz file
    # print('\nWrite to npz files:')
    # results = saveToNpzFile(ProcPath, train_imgs, train_lbls, valid_imgs, valid_lbls)

    # print('\nNPZ files:')
    # for f_abspath in results:
    #     outstr = 'exist.' if exists(f_abspath) else 'not exist...'
    #     print("\n'{}' {}".format(f_abspath, outstr))
    #     showNpzFile(f_abspath)

if __name__=='__main__':
    print('torch: ', torch.__version__)
    print('torchvision: ', torchvision.__version__)
    main()
