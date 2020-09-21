# From: https://www.kaggle.com/c/dog-breed-identification/data
# Author: Morpheus Hsieh

from __future__ import print_function, division

import os, sys
import argparse
import copy
import io
import json
import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import join, isfile

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
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()






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


    return 


if __name__=='__main__':
    main()

