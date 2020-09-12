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
    params['RootPath'] = cfg.WORK.PATH
    params['RawDataPath'] = cfg.DATA.PATH_RAW
    params['TestImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TEST)
    params['TrainImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TRAIN)
    params['CsvLabels'] = join(cfg.DATA.PATH_RAW, cfg.DATA.CSV_LABELS)
    params['ProcessedBreeds'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.CSV_BREEDS)
    params['ProcessedLabels'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.CSV_LABELS)
    params['TrainData'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.TRAIN_DATA)
    params['ValidData'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.VALID_DATA)
    params['PretrainedModel'] = join(cfg.PRETRAINED.PATH, cfg.PRETRAINED.FNAME)
    params['FracForTrain'] = cfg.TRAIN.FRAC_FOR_TRAIN
    params['NumBreedClasses'] = cfg.TRAIN.NUM_BREED_CLASSES
    params['BatchSize'] = cfg.TRAIN.BATCH_SIZE
    params['LearningRate'] = cfg.TRAIN.LEARNING_RATE
    params['NumEpochs'] = cfg.TRAIN.NUM_EPOCHS
    return params


def loadMostPopularBreeds(num=16):
    pass


def main():
    print("\nPyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    args = parse_args()
    print(args)

    CFG = get_cfg_defaults()
    CFG.merge_from_file(args.cfg)
    CFG.freeze()
    print('\n', CFG)

    Params = getParams(CFG)
    print('\nParameters:'); print(json.dumps(Params, indent=4))



    # # Load pretrained model
    # pre_model_path = join(ModelPath, 'resnet50-19c8e357.pth')
    # pre_model_wts = torch.load(pre_model_path)
    # resnet.load_state_dict(pre_model_wts)


if __name__=='__main__':
    main()


