from __future__ import print_function, division

import os, sys
from os.path import abspath, dirname, isfile, join
parent_path = abspath(dirname(dirname(__file__)))
if parent_path not in sys.path: sys.path.insert(0, parent_path)

import argparse
import copy
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from io import StringIO
from os import listdir
from os.path import join, exists
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, utils

from configs.config_train import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Dog Breed identification')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()

def printInfoHead(df, num=10, title=None, newLine=True):
    if newLine: print()
    if title is not None: print(f'{title} info and head:')
    print(df.info())
    print(df.head(num))
    return

def printInfo(df, title=None, newLine=True, indent=2):
    buf = StringIO()
    df.info(buf=buf)
    pad_str = (' ' * indent)
    old_str = '\n'
    new_str = '\n' + pad_str
    outstr = pad_str + buf.getvalue().replace(old_str, new_str)

    if newLine: print()
    if title is not None: print(f'{title} information:')
    print(outstr)
    return

def printHead(df, num=4, title=None, newLine=True, indent=2):
    if newLine: print()
    if title is not None: print(f'{title} info and head:')
    outstr = df.head(num).to_string()
    pad_str = (' ' * indent)
    old_str = '\n'
    new_str = '\n' + pad_str
    outstr = pad_str + outstr.replace(old_str, new_str)
    print(outstr)
    return


# Read breed information from csv
def getCsvLabels(cfg):
    path = cfg.DATA.PATH
    fname = cfg.DATA.FNAME_LABELS
    df = pd.read_csv(join(path, fname))
    return df


def main():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Read arguments
    args = parse_args()
    print(args)

    # Read configurations 
    CFG = get_cfg_defaults()
    CFG.merge_from_file(args.cfg)
    CFG.freeze()
    print('\n', CFG)

    # Get Labels information from csv
    df_labels = getCsvLabels(CFG)
    printInfo(df_labels, title='Labels')
    printHead(df_labels, title='Labels')

    return



if __name__=='__main__':
    main()


