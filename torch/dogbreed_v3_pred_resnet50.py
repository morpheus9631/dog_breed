# Author: Morpheus (morpheus.hsieh@gmail.com)

from __future__ import print_function, division

import os, sys
from os.path import dirname

curr_path = os.path.abspath(dirname(__file__))
if curr_path not in sys.path: sys.path.append(curr_path)

parent_path = os.path.abspath(dirname(curr_path))
if parent_path not in sys.path: sys.path.append(parent_path)

import argparse
import copy
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import datetime
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import join, isfile, split, exists
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms, utils

from configs.config_train_v3 import get_cfg_defaults
from utils.myPrint import printHead, printInfo, printList, printDict


# Read parameters from yaml file
def parse_args():
    parser = argparse.ArgumentParser(description='Dog Breed identification')
    parser.add_argument("--cfg", type=str, default="configs/config_train_v3.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


# Read breed information from csv
def getCsvLabels(cfg):
    path = cfg.DATA.PATH
    fname = cfg.DATA.FNAME_LABELS
    df = pd.read_csv(join(path, fname))
    return df


# Get seletced breeds
def getSelectedBreeds(df, numClasses=0):
    df1 = df.groupby("breed")["id"].count().reset_index(name="count")
    df1 = df1.sort_values(by='count', ascending=False).reset_index(drop=True)
    df1.insert(0, 'breed_id', df1.index)
    if numClasses > 0:
        df1 = df1.head(numClasses)
    selected_breeds = df1['breed'].tolist()
    return df1, selected_breeds


# Convert list to dictrionary
def df2dict(df):
    # dict_fw = dict(df[['breed', 'breed_id']].values)
    dict_bw = dict(df[['breed_id', 'breed']].values)
    return dict_bw


class myDataset(Dataset):

    def __init__(self, path, transform=None):
        
        iid_list = [
            f.replace('.jpg', '') for f in listdir(path) \
            if f.endswith('.jpg') and isfile(join(path, f))
        ]

        self.len = len(iid_list)
        self.images = iid_list
        self.transform = transform
        self.path = path

    def __getitem__(self, index):
        iid = self.images[index] 
        img = join(self.path, iid) + '.jpg'
        img_pil = Image.open(img)

        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        
        return [img_tensor, iid]

    def __len__(self):
        return self.len    


def createDataLoader(cfg):
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    test_path = cfg.DATA.PATH_TEST
    dset = myDataset(test_path, transform=transform)
    dsize = len(dset)

    batch_size = cfg.PREDICT.BATCH_SIZE
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    return dloader


def printDataLoader(loader):
    imgs, iids = next(iter(loader))
    print('\nImage shape:', imgs.shape)

    print('\nImage ids')
    id_list = [''.join(iid) for iid in iids]
    print('  '+'\n  '.join(id_list))

    img = imgs[0]
    print('\nImage shape:', img.shape)

    print('\nImage tensor:')
    print(img)
    return


# Build Model
def buildModel(cfg, numClasses, use_gpu):
    model = models.resnet50(pretrained=True)

    # freeze all model parameters
    for param in model.parameters():
        model.requires_grad = False

    # new final layer with 16 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numClasses)
    
    path = cfg.PREDICT.MODEL_PATH
    fname = cfg.PREDICT.MODEL_FILE
    model_wts = join(path, fname)
    model.load_state_dict(torch.load(model_wts))
        
    if use_gpu: model = model.cuda()
    return model


# Predict model
def predict(selected_bds, model, dloader, dict_bid_bw):
    cols_preds = ['id', 'prediction']
    df_preds = pd.DataFrame(columns=cols_preds)

    cols_probs = ['id'] + selected_bds
    df_probs = pd.DataFrame(columns=cols_probs)

    start_time = time.time()
    print('\nStart predicting...')

    model.eval()

    for i, (inputs, iids) in enumerate(dloader):
        inputs = Variable(inputs.cuda())
        iid_list = list(iids)
    
        # with torch.set_grad_enabled(True):
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            probs = nn.functional.softmax(outputs, dim=1)
        
            if i == 0:
                print('\nProbs:', probs.shape); # print(probs)
                print('\nPreds:', preds.shape); # print(preds)
                print()
    
        pred_list = preds.tolist()
        pred_breeds = [dict_bid_bw.get(x) for x in pred_list]
    
        df_tmp = pd.DataFrame({
            'id': iid_list,
            'prediction': pred_breeds
        })
        df_preds = df_preds.append(df_tmp)

        df_tmp = pd.DataFrame({'id': iid_list})
        df_tmp[selected_bds] = pd.DataFrame(probs.tolist())
        df_probs = df_probs.append(df_tmp)

        print(i, end=', ')
    print()
    print('Testing time: {:10f} minutes'.format((time.time()-start_time)/60))
    
    return df_preds, df_probs


# Save prediction results to csv file
def saveToCsv(out_path, preds, probs):
    currDT = datetime.now()
    currStr = currDT.strftime("%Y%m%d-%H%M%S")

    fname1 = 'Prediction_{}.csv'.format(currStr)
    preds.to_csv(join(out_path, fname1), index=False)

    fname2 = 'Probability_{}.csv'.format(currStr)
    probs.to_csv(join(out_path, fname2), index=False)

    print("\n'{} and '{}' have been saved.".format(fname1, fname2))
    return 


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

    # Get selected breeds, 0 is all breeds
    df_breeds, selected_breeds = getSelectedBreeds(df_labels)
    printList(selected_breeds)

    NumClasses = CFG.TRAIN.NUM_CLASSES
    if NumClasses == 0: NumClasses = len(selected_breeds)
    print('\nSelected breeds:', NumClasses)

    # Convert list to dictionary
    dict_bid_bw = df2dict(df_breeds)
    print('\nBackward breed dict:'); printDict(dict_bid_bw)

    # Create dataloader
    dloader = createDataLoader(CFG)
    printDataLoader(dloader)

    # Use GPU for train
    torch.cuda.empty_cache() # release all gpuy memory
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('\nGPU:', device)

    # Build Model
    model = buildModel(CFG, NumClasses, use_gpu)
    print(model)

    # Predict
    df_preds, df_probs = predict(
        selected_breeds, model, dloader, dict_bid_bw
    )

    # Save prediction results
    OutPath = CFG.OUTPUT.PATH
    saveToCsv(OutPath, df_preds, df_probs)

    return

if __name__=='__main__':
    main()
