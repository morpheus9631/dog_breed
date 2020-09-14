# Kaggle: https://www.kaggle.com/c/dog-breed-identification/data
# Author: Morpheus Hsieh

from __future__ import print_function, division

import argparse
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import join
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms, utils

from configs.config_train import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()

def getParams(cfg):
    params = {}
    params['BatchSize'] = cfg.TRAIN.BATCH_SIZE
    # params['CsvLabels'] = join(cfg.DATA.PATH_RAW, cfg.DATA.CSV_LABELS)
    params['FracForTrain'] = cfg.TRAIN.FRAC_FOR_TRAIN
    params['LearningRate'] = cfg.TRAIN.LEARNING_RATE
    params['NumPopularClasses'] = cfg.TRAIN.NUM_POPULAR_CLASSES
    params['NumEpochs'] = cfg.TRAIN.NUM_EPOCHS
    params['PretrainedModel'] = join(cfg.PRETRAINED.PATH, cfg.PRETRAINED.FNAME)
    params['ProcessedBreeds'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.CSV_BREEDS)
    params['ProcessedLabels'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.CSV_LABELS)
    params['RawDataPath'] = cfg.DATA.PATH_RAW
    params['RootPath'] = cfg.WORK.PATH
    params['TestImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TEST)
    params['TrainImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TRAIN)
    params['TrainData'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.TRAIN_DATA)
    params['ValidData'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.VALID_DATA)
    return params

def getMostPopularBreeds(df, numClasses=16):
    selected_breed_list = list(df['breed'][:numClasses] )
    return (selected_breed_list)

def df2dict(df, direc='fw'):
    dic = {}
    for i, row in df.iterrows():
        if direc == 'fw':   # fw = forward
            dic[row['breed']] = row['breed_id']
        elif direc == 'bw': # bw = backward
            key = str(row['breed_id'])
            dic[key] = row['breed']
    return dic

def main():
    print("\nPyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Read arguments
    args = parse_args()
    print(args)

    # Read configurations 
    CFG = get_cfg_defaults()
    CFG.merge_from_file(args.cfg)
    CFG.freeze()
    print('\n', CFG)

    Params = getParams(CFG)
    print('\nParameters:'); print(json.dumps(Params, indent=2))

    # Read breed information from precessed breed csv file
    csv_proc_breeds = Params['ProcessedBreeds']
    df_breeds = pd.read_csv(csv_proc_breeds)
    print('\nBreeds info:', df_breeds.info())
    print('\nBreeds head:', df_breeds.head())

    # Get most popular breeds
    NumClasses = Params['NumPopularClasses']
    selected_breeds = getMostPopularBreeds(df_breeds, NumClasses)
    print('\nSelected breeds: [\n  {}\n]'.format('\n  '.join(selected_breeds)))

    df_selected_breeds = df_breeds[df_breeds['breed'].isin(selected_breeds)]

    # Build breed dictionary, both forward and backward
    breed_dic_fw = df2dict(df_selected_breeds)
    print('\nBreed dict (forward):')
    print(json.dumps(breed_dic_fw, indent=2))

    breed_dic_bw = df2dict(df_selected_breeds, 'bw')
    print('\nBreed dict (backward):')
    print(json.dumps(breed_dic_bw, indent=2))

    # Read labels information from csv file
    csv_prco_labels = Params['ProcessedLabels']
    df_labels = pd.read_csv(csv_prco_labels)
    print('\nLabels info:', df_labels.info())
    print('\nLabels head:', df_labels.head())

    selected_data = df_labels[df_labels['breed'].isin(selected_breeds)]
    print('\nSelected labels:');print(selected_data[:10])





    # # Load pretrained model
    # pre_model_path = join(ModelPath, 'resnet50-19c8e357.pth')
    # pre_model_wts = torch.load(pre_model_path)
    # resnet.load_state_dict(pre_model_wts)


if __name__=='__main__':
    main()


