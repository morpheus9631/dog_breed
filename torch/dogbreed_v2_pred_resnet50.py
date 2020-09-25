# From: https://www.kaggle.com/c/dog-breed-identification/data
# Author: Morpheus Hsieh

from __future__ import print_function, division

import os, sys
from os.path import abspath, dirname

curr_path = abspath(dirname(__file__))
parent_path = abspath(dirname(curr_path))
if parent_path not in sys.path: sys.path.insert(0, parent_path)

import argparse
import copy
import io
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join, exists
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, utils

from configs.config_train_v2 import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Dog Breed identification by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train_v2.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


# Read breed information from csv
def getBreedDataframe(path, fname):
    f_abspath = join(path, fname)
    df_breeds = pd.read_csv(f_abspath)
    return df_breeds


def getMostPopularBreeds(df, numClasses=16):
    df1 = df.sort_values(['count', 'breed'], ascending=(False, True))
    df1 = df1.head(numClasses)
    return df1


def df2dict(df, dire='forward'):
    dic = {}
    for i, row in df.iterrows():
        if dire == 'forward':
            dic[row['breed']] = row['breed_id']
        elif dire == 'reverse':
            dic[row['breed_id']] = row['breed']
    return dic


def prettyPrint(d, indent=0):
    print('{')
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key))
            prettyPrint(value, indent+1)
        else:
            print('  ' * (indent+1) + f"{key}: {value}")
    print('}')


def dfInfo2Str(df, indent=4):
    buf = io.StringIO()
    df.info(buf=buf)
    pad_str = (' ' * indent)
    old_str = '\n'
    new_str = '\n' + pad_str
    outstr = buf.getvalue().replace(old_str, new_str)
    return pad_str + outstr


class myDataset(Dataset):

    def __init__(self, path, df, transform=None):

        self.images = list(df['image'])
        self.labels = list(df['breed_id'])
        self.len = len(self.images)
        self.path = path

        self.transform = transform

    def __getitem__(self, index):
        iid = self.images[index]
        f_abspath = join(self.path, iid) + '.jpg'
        img_pil = Image.open(f_abspath)

        if self.transform is not None:
            img = self.transform(img_pil)

        lbl = int(self.labels[index])
        
        return [img, lbl, iid]

    def __len__(self):
        return self.len


def getDataSet(path, df):
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    dataSet = myDataset(path, df, transform=transform)
    return dataSet
    

def getModel(numClasses, preTranPath, preTranModel):
    model = models.resnet50(pretrained=True)

    # freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # New final layer with NumClasses
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, numClasses)

    # load pretrained mode
    f_abspath = join(preTranPath, preTranModel)
    if exists(f_abspath) and isfile(f_abspath):
        model.load_state_dict(torch.load(f_abspath))
    return model


def predict(model, selectedBreeds, loader, dictBreed):
    cols_preds = ['id', 'prediction']
    df_preds = pd.DataFrame(columns=cols_preds)

    cols_probs = ['id'] + selectedBreeds
    df_probs = pd.DataFrame(columns=cols_probs)

    start_time = time.time()
    print('\nStart testing...')

    model.eval()

    for i, (inputs, labels, iids) in enumerate(loader):

        inputs = Variable(inputs.cuda())
        iid_list = list(iids)
    
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        if i == 0:
            print('\nImage id list: [\n  {}\n]'.format('\n  '.join(iid_list)))
            print(); print(probs.shape); print(type(preds)); print(probs)
            print(); print(preds.shape); print(preds)
            print()
    
        pred_list = preds.tolist()
        pred_breeds = [dictBreed.get(x) for x in pred_list]
    
        df_tmp = pd.DataFrame({
            'id': iid_list,
            'prediction': pred_breeds
        })
        df_preds = df_preds.append(df_tmp)

        df_tmp = pd.DataFrame({'id': iid_list})
        df_tmp[selectedBreeds] = pd.DataFrame(probs.tolist())
        df_probs = df_probs.append(df_tmp)

        print(i, end=', ')
    print()    
    print('Testing time: {:10f} minutes'.format((time.time()-start_time)/60))    

    return  df_preds, df_probs


def saveToCsv(outPath, dfPreds, dfProbs):
    currDT = datetime.now()
    currStr = currDT.strftime("%Y%m%d-%H%M%S")

    fname = 'Prediction_{}.csv'.format(currStr)
    dfPreds.to_csv(join(outPath, fname), index=False)

    fname = 'Probaility_{}.csv'.format(currStr)
    dfProbs.to_csv(join(outPath, fname), index=False)
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

    # Get breed info from csv file
    ProcPath = CFG.PROCESSED.PATH
    CsvBreed = CFG.PROCESSED.FNAME_BREEDS + '.csv'
    df_breeds = getBreedDataframe(ProcPath, CsvBreed)
    print(); print(df_breeds.info())
    print(); print(df_breeds.head())

    # Get most popular breeds
    NumClasses = CFG.TRAIN.NUM_CLASSES
    df_breeds_selected = getMostPopularBreeds(df_breeds, NumClasses)

    selected_brds = list(df_breeds_selected['breed'])
    print('\nSelected breeds: [\n  {}\n]'.format('\n  '.join(selected_brds)))

    selected_bids = list(df_breeds_selected['breed_id'])
    print('\nSelected breed ids:\n  {}'.format(selected_bids))

    # Build breed dictionaries
    dict_breed_fw = df2dict(df_breeds_selected)
    dict_breed_rv = df2dict(df_breeds_selected, 'reverse')

    print('\nBreeds dict forward:'); 
    print(json.dumps(dict_breed_fw, indent=2))

    print('\nBreeds dict reverse:'); 
    prettyPrint(dict_breed_rv)

    # Selected labels
    CsvLabelsProc = CFG.PROCESSED.FNAME_LABELS + '.csv'
    f_abspath = join(ProcPath, CsvLabelsProc)

    df_labels = pd.read_csv(f_abspath)
    print('\nOrigin labels:\n')
    print(dfInfo2Str(df_labels))

    df_labels_selected = df_labels[df_labels['breed_id'].isin(selected_bids)]

    print('\nSelected labels:\n')
    print(dfInfo2Str(df_labels_selected))

    print('\nSelected labels Head:')
    print(df_labels_selected.head())

    # Set dataloader
    BatchSize = CFG.TRAIN.BATCH_SIZE
    TrainPath = CFG.DATA.PATH_TRAIN
    dataSet = getDataSet(TrainPath, df_labels_selected)
    dataLoader = DataLoader(dataSet, batch_size=BatchSize, shuffle=True)
    dataSize = len(dataSet)

    imgs, lbls, iids = next(iter(dataLoader))
    print('\nImage type:', type(imgs))
    print('      size: ', imgs.size())

    print('\nLabel type:', type(lbls))
    print('      size: ', lbls.size())

    print('\nImage ids:')
    id_list = [''.join(iid) for iid in iids]
    print('  '+'\n  '.join(id_list))

    img = imgs[0]
    print('\nImage shape:', img.shape)
    # print(); print(img)
    print('\nLabels:', lbls)

    # Use GPU for train
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(); print(device)

    # Build Model
    PretranPath = CFG.PRETRAINED.PATH
    PretranFile = CFG.PRETRAINED.FNAME_PREMODEL
    model = getModel(NumClasses, PretranPath, PretranFile)
    # print(); print(model)

    if use_gpu: model = model.cuda()

    df_preds, df_probs = predict(
        model, selected_brds, dataLoader, dict_breed_rv
    )
    
    print(); print(df_preds.info())
    print(); print(df_preds.head())

    print(); print(df_probs.info())
    print(); print(df_probs.head())

    # Save predict results to csv file
    OutPath = CFG.OUTPUT.PATH
    saveToCsv(OutPath, df_preds, df_probs)

    return 


if __name__=='__main__':
    main()

