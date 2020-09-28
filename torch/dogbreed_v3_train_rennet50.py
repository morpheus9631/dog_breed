from __future__ import print_function, division

import os, sys
from os.path import dirname

curr_path = os.path.abspath(dirname(__file__))
parent_path = os.path.abspath(dirname(curr_path))
if parent_path not in sys.path: sys.path.insert(0, parent_path)
# print(parent_path)

import argparse
import copy
import csv
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from io import StringIO
from os import listdir
from os.path import join, exists, isfile
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, utils

from configs.config_train_v3 import get_cfg_defaults


# Read parameters from yaml file
def parse_args():
    parser = argparse.ArgumentParser(description='Dog Breed identification')
    parser.add_argument("--cfg", type=str, default="../configs/config_train_v3.yaml",
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
        
        # get random image id without duplicates
        idx_ary = np.arange(len(df))
        idx_ary_rand = np.random.permutation(idx_ary) # random shuffle
        
        data_len = train_len if phase=='train' else valid_len
        data = df.loc[idx_ary_rand[:data_len]]

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
def currDatetimeStr():
    currDT = datetime.now()
    currStr = currDT.strftime("%Y%m%d-%H%M")
    return currStr


# Train and validate Model
def train_model(
        cfg, loader, model, criterion, optimizer, scheduler, num_epochs=3
    ):
    print('\nStart training...')
    since = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_valid_acc = 0.0
    best_train_acc = 0.0
    
    # Write accurancy and loss to csv fiel
    outPath = cfg.OUTPUT.PATH
    currStr = currDatetimeStr()
    fname_out = 'output_{}.csv'.format(currStr)
    abspath_out = join(outPath, fname_out)
    columns = [
        'epoch', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss'
    ]

    csvFile = open(abspath_out, 'w', newline='')
    csvWrite = csv.writer(csvFile)
    csvWrite.writerow(columns)
    csvFile.flush()
    
    # Dataset sizes
    dataset_sizes = {
        'train': len(loader['train'].dataset),
        'valid': len(loader['valid'].dataset)
    }

    epoch_maxlen = len(str(num_epochs))

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
                
                # forward, 
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
                train_loss = running_loss / data_size
                train_acc  = running_corrects / data_size
            else:
                valid_loss = running_loss / data_size
                valid_acc  = running_corrects / data_size

            # update best valid accuracy and best model 
            if phase == 'valid' and valid_acc >= best_valid_acc:
                best_train_acc = train_acc
                best_valid_acc = valid_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # save best model
                best_train_acc_str = int(best_train_acc * 10000)
                best_valid_acc_str = int(best_valid_acc * 10000)
                fname_best_model = 'resnet50_{}_t{}_v{}.pth'.format(
                    currStr, best_train_acc_str, best_valid_acc_str
                )
                abspath_best_model = join(outPath, fname_best_model)
                torch.save(best_model_wts, abspath_best_model)
                currStr = currDatetimeStr()

        fmt_str = "Epoch [{:"+f'{epoch_maxlen}'+"d}/{:"+f'{epoch_maxlen}'+"d}] " \
                + "train loss: {:.4f} acc: {:.4f}, valid loss: {:.4f} acc: {:.4f}"
        print(fmt_str.format(
            epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc
        ))

        out_vals = [ 
            epoch, train_acc.item(), train_loss, valid_acc.item(), valid_loss
        ]
        csvWrite.writerow(out_vals)
        csvFile.flush()

    csvFile.close()
    print('\nBest val Acc: {:4f}'.format(best_valid_acc))

    model.load_state_dict(best_model_wts)
    return model, best_valid_acc


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
    NumClasses = CFG.TRAIN.NUM_CLASSES
    df_breeds, selected_breeds = getSelectedBreeds(df_labels)
    if NumClasses == 0: NumClasses = len(selected_breeds)
    print('\nSelected breeds:', NumClasses)
    printList(selected_breeds)

    # Convert list to dictionary
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
    model = buildModel(CFG, NumClasses, use_gpu)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training and validate
    NumEpochs = CFG.TRAIN.NUM_EPOCHS
    start_time = time.time()
    best_model, best_valid_acc = train_model(
        CFG, dataloader, model, criterion, optimizer, exp_lr_scheduler, NumEpochs
    )
    print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

    return (0)


if __name__=='__main__':
    main()


