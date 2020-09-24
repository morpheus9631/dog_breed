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


# Read parameters from yaml file
def parse_args():
    parser = argparse.ArgumentParser(description='Dog Breed identification')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


# Print pandas dataframe information
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


# Print pandas dataframe first N rows data
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


# Get seletced breeds
def getSelectedBreeds(df, numClasses=0):
    df1 = df.groupby("breed")["id"].count().reset_index(name="count")
    df1 = df1.sort_values(by='count', ascending=False).reset_index(drop=True)
    df1.insert(0, 'breed_id', df1.index)
    if numClasses > 0:
        df1 = df1.head(numClasses)
    selected_breeds = df1['breed'].tolist()
    return df1, selected_breeds


# Pretty Print list
def printList(array, indent=2):
    isStr = isinstance(array[0], str)
    padstr = ' ' * indent
    outstr = ''
    for e in array:
        if len(outstr) > 0: outstr += ',\n'
        if isinstance(e, str):
            outstr += padstr + f"'{e}'"
        else:
            outstr += padstr + f'{e}'
    outstr = '[\n{}\n]'.format(outstr)
    print(outstr)
    return 


# Convert list to dictrionary
def df2dict(df):
    dict_fw = dict(df[['breed', 'breed_id']].values)
    dict_bw = dict(df[['breed_id', 'breed']].values)
    return dict_fw, dict_bw


# Pretty print dictionary
def printDict(dic, indent=2):
    array = []
    key_maxlen = 0
    item_cnt = 0
    item_size = len(dic)
    split_str = ': '
    
    isExtLen = []
    for key, val in dic.items():
        if key_maxlen < len(str(key)): 
            key_maxlen = len(str(key))
        
        isStrKey = isinstance(key, str)
        isExtLen.append(2 if isStrKey else 1)

        tmpstr = ''
        tmpstr += f"'{key}'" if isStrKey else f"{key}"
        tmpstr += split_str
        tmpstr += f"'{val}'" if isinstance(val, str) else f"{val}"

        item_cnt += 1
        if item_cnt < item_size: tmpstr += ','
        array.append(tmpstr)

    for i in range(len(array)):
        inStr = array[i]
        ary = inStr.split(split_str)
        key = ary[0].ljust(key_maxlen + isExtLen[i])
        val = ary[1]
        array[i] = (' '*indent) + key + split_str + val
        
    outstr = '{\n' + '\n'.join(array) + '\n}'
    print(outstr)
    return


# Build processed dataframe
def getBidDataframe(path, df_src, dic):
    df_dst = pd.DataFrame(columns=['image', 'breed_id'])
    df_dst['breed_id'] = df_src.breed.map(dic)

    df_dst['image'] = df_src.apply (
        lambda row: row['id'] \
        if exists(join(path, row['id']+'.jpg')) else None, 
        axis=1
    )
    return df_dst


# My Dataset
class myDataset(Dataset):

    def __init__(self, df, path, phase='train', frac=0.8, transform=None):
        
        num_rows = df.shape[0]
        train_len = int(float(frac) * float(num_rows))
        valid_len = num_rows - train_len
        
        data = df.head(train_len) if phase=='train' else df.tail(valid_len)
        self.images = data['image'].tolist()
        self.labels = data['breed_id'].tolist()

        self.transform = transform
        self.len = len(self.images)
        self.path = path

    def __getitem__(self, index):
        iid = self.images[index]
        f_abspath = join(self.path, iid+'.jpg')
        img_pil = Image.open(f_abspath)

        if self.transform is not None:
            img = self.transform(img_pil)

        lbl = int(self.labels[index])
        return [img, lbl, iid]

    def __len__(self):
        return self.len


# Create dataloader
def getDataLoader(cfg, df):
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    phases = ['train', 'valid']
    frac = cfg.TRAIN.FRAC_FOR_TRAIN
    path = cfg.DATA.PATH_TRAIN
    dset = { 
        x: myDataset(df, path, phase=x, frac=frac, transform=transform) \
        for x in phases 
    }

    BatchSize = cfg.TRAIN.BATCH_SIZE
    dloader = {
        x: DataLoader(dset[x], batch_size=BatchSize, shuffle=True) \
        for x in phases
    }

    dsizes = { x: len(dset[x]) for x in phases }
    return dloader


# Print data loader information
def printDataLoaderInfo(loader):
    train_loader = loader['train']
    imgs, lbls, iids = next(iter(train_loader))
    print('\nImage shape:', imgs.size())
    print('Label shape:', lbls.size())
    
    print('\nImage iid:')
    id_list = [''.join(iid) for iid in iids]
    print('  '+'\n  '.join(id_list))
    
    img = imgs[0]
    print('\nImage shape:', img.shape)
    print(); print(img)
    print('\nLabels:', lbls[0])
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
    
    path = cfg.PRETRAINED.PATH
    fname = cfg.PRETRAINED.FNAME_PREMODEL
    f_abspath = join(path, fname)
    if exists(f_abspath) and isfile(f_abspath):
        pretrain_model = torch.load(f_abspath)
        model.load_state_dict(pretrain_model)
        
    if use_gpu: 
        model = model.cuda()
        
    return model


# Current datetime to string
def currDTstr():
    currDT = datetime.now()
    currStr = currDT.strftime("%Y%m%d-%H%M%S")
    return currStr


# Train and validate Model
def train_model(loader, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history  = {
        'train_acc': [],
        'train_los': [],
        'valid_acc': [],
        'valid_los': []
    }
    
    dataset_sizes = {
        'train': len(loader['train'].dataset),
        'valid': len(loader['valid'].dataset)
    }

    for epoch in range(num_epochs):
        
        for phase in ['train', 'valid']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels, iids in loader[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistic
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
            
            data_size = dataset_sizes[phase]
            
            if phase == 'train':
                train_epoch_loss = running_loss / data_size
                train_epoch_acc  = running_corrects / data_size
            else:
                valid_epoch_loss = running_loss / data_size
                valid_epoch_acc  = running_corrects / data_size

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        history['train_acc'].append(train_epoch_acc.item())
        history['train_los'].append(train_epoch_loss)
        history['valid_acc'].append(valid_epoch_acc.item())
        history['valid_los'].append(valid_epoch_loss)
        
        print('Epoch [{:3d}/{:3d}] train loss: {:.4f} acc: {:.4f}' 
              '\n                valid loss: {:.4f} acc: {:.4f}'.format(
                  epoch, num_epochs - 1,
                  train_epoch_loss, train_epoch_acc, 
                  valid_epoch_loss, valid_epoch_acc))

    print('\nBest val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, best_acc, history


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
    print('\nSelected breeds:')
    printList(selected_breeds)

    # Conver list to dictionary
    dict_bid_fw, dict_bid_bw = df2dict(df_breeds)
    print('\nForward breed dict:');  printDict(dict_bid_fw)
    # print('\nBackward breed dict:'); printDict(dict_bid_bw)

    # Build processed labels file
    TrainPath = CFG.DATA.PATH_TRAIN
    df_data = getBidDataframe(TrainPath, df_labels, dict_bid_fw)
    printInfo(df_data, title='Breed ids')
    printHead(df_data, title='Breed ids')

    # Create dataset and dataloader
    dataloader = getDataLoader(CFG, df_data)
    printDataLoaderInfo(dataloader)

    # Use GPU
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print('\n'); print(device)

    # Build model 
    NumClasses = len(selected_breeds)
    model = buildModel(CFG, NumClasses, use_gpu)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training and validate
    NumEpochs = CFG.TRAIN.NUM_EPOCHS
    start_time = time.time()
    best_model, best_acc, history = train_model(
        dataloader, model, criterion, optimizer, exp_lr_scheduler, NumEpochs
    )
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))


    return



if __name__=='__main__':
    main()


