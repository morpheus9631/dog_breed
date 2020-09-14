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
    params['FracForTrain'] = cfg.TRAIN.FRAC_FOR_TRAIN
    params['NumClasses'] = cfg.TRAIN.NUM_CLASSES
    params['PretrainedModel'] = join(cfg.PRETRAINED.PATH, cfg.PRETRAINED.FNAME)
    # params['ProcessedPath'] = cfg.PROCESSED.PATH
    params['ProcessedBreeds'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.FNAME_BREEDS+'.npz')
    params['ProcessedLabels'] = join(cfg.PROCESSED.PATH, cfg.PROCESSED.FNAME_LABELS+'.npz')
    params['LearningRate'] = cfg.TRAIN.LEARNING_RATE
    params['Momentum'] = cfg.TRAIN.MOMENTUM
    params['StepSize'] = cfg.TRAIN.STEP_SIZE
    params['Gamma'] = cfg.TRAIN.GAMMA
    params['NumEpochs'] = cfg.TRAIN.NUM_EPOCHS
    # params['RawDataPath'] = cfg.DATA.PATH_RAW
    # params['RootPath'] = cfg.WORK.PATH
    # params['TestImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TEST)
    # params['TrainImgPath'] = join(cfg.DATA.PATH_RAW, cfg.DATA.DIR_TRAIN)
    return params

def readNpzFile(f_abspath):
    load_data = np.load(join(f_abspath), allow_pickle=True)
        
    col_names = load_data.files
    df = pd.DataFrame.from_dict(
        {col: load_data[col] for col in col_names}
    )
    df.columns = col_names
    return df

def getMostPopularBreeds(df, numClasses=16):
    selected_breeds = list(df['breed'][:numClasses] )
    selected_breed_ids = list(df['breed_id'][:numClasses] )
    return selected_breeds, selected_breed_ids

def df2dict(df, direc='fw'):
    dic = {}
    for i, row in df.iterrows():
        if direc == 'fw':   # fw = forward
            dic[row['breed']] = row['breed_id']
        elif direc == 'bw': # bw = backward
            key = str(row['breed_id'])
            dic[key] = row['breed']
    return dic

def getTrainAndValidData(selectedData, fracForTrain=0.8):
    img_list = list(selectedData['image'])
    lbl_list = list(selectedData['breed_id'])

    num_rows = len(img_list)
    train_len = int(float(fracForTrain) * float(num_rows))

    x_imgs = img_list[:train_len]
    x_lbls = lbl_list[:train_len]

    y_imgs = img_list[train_len:]
    y_lbls = lbl_list[train_len:]

    return {
        'train': [ x_imgs, x_lbls ],
        'valid':  [y_imgs, y_lbls ]
    } 

class myDataset(Dataset):

    Fname = 'labels_processed.npz'

    def __init__(self, f_abspath, selected_bids, transform=None):
        fname = self.Fname
        load_data = np.load(f_abspath, allow_pickle=True)
        
        col_names = list(load_data.files)
        df = pd.DataFrame.from_dict(
            {col: load_data[col] for col in col_names}
        )
        # print(col_names)
        df.columns = col_names

        df_selected = df[df['breed_id'].isin(selected_bids)]
        # print(df_selected.info())
        # print(df_selected.head())

        self.images = df_selected['image']
        self.labels = df_selected['breed_id']

        self.transform = transform
        self.len = len(df_selected)

    def __getitem__(self, index):
        img_path = self.images[index]
        img_pil = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img_pil)

        lbl = int(self.labels[index])
        return [img, lbl]

    def __len__(self):
        return self.len


def train_model(dataloders, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    dataset_sizes = {'train': len(dataloders['train'].dataset), 
                     'valid': len(dataloders['valid'].dataset)}

    for epoch in range(num_epochs):
        
        for phase in ['train', 'valid']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloders[phase]:
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
            
            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch [{}/{}] train loss: {:.4f}, acc: {:.4f}' 
              ',\n            valid loss: {:.4f}, acc: {:.4f}'.format(
                  epoch, num_epochs - 1,
                  train_epoch_loss, train_epoch_acc, 
                  valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


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

    # # Read breed information from precessed breed file
    npz_breeds = Params['ProcessedBreeds']
    df_breeds = readNpzFile(npz_breeds)
    print('\nBreeds info:'); print(df_breeds.info())
    print('\nBreeds head:'); print(df_breeds.head())

    # Get most popular breeds
    NumClasses = Params['NumClasses']
    selected_breeds, selected_bids = getMostPopularBreeds(df_breeds, NumClasses)
    print('\nSelected breeds: [\n  {}\n]'.format('\n  '.join(selected_breeds)))
    print('\nSelected breed ids:\n  {}'.format(selected_bids))

    df_selected_breeds = df_breeds[df_breeds['breed'].isin(selected_breeds)]

    # Build breed dictionary, both forward and backward
    breed_dic_fw = df2dict(df_selected_breeds)
    print('\nBreed dict (forward):')
    print(json.dumps(breed_dic_fw, indent=2))

    breed_dic_bw = df2dict(df_selected_breeds, 'bw')
    print('\nBreed dict (backward):')
    print(json.dumps(breed_dic_bw, indent=2))

    # Normalize
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    # Build dataset
    npz_labels = Params['ProcessedLabels']
    trainSet = myDataset(npz_labels, selected_bids, transform=transform)
    # validSet = myDataset(npz_labels, selected_bids, transform=transform)

    # BatchSize = Params['BatchSize']
    # print('\nBatch size:', BatchSize)

    # trainLoader = DataLoader(trainSet, batch_size=BatchSize, shuffle=True)
    # validLoader = DataLoader(validSet, batch_size=BatchSize, shuffle=False)

    # imgs, lbls = next(iter(trainLoader))
    # print('\nImages size: ', imgs.size())
    # print('Labels size: ', lbls.size())
    # print('\nImage shape:', imgs[0].shape)
    # print('Label shape:', lbls[0].shape)

    # dataset_sizes = { 'train': len(trainSet), 'valid': len(validSet) }
    # print('\nDataset sizes:', dataset_sizes)

    # # Use GPU
    # use_gpu = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_gpu else "cpu")
    # print(); print(device)

    # # Set Models
    # resnet = models.resnet50(pretrained=True)

    # # # Load pretrained model
    # pre_model_path = Params['PretrainedModel']
    # pre_model_wts = torch.load(pre_model_path)
    # resnet.load_state_dict(pre_model_wts)

    # # freeze all model parameters
    # for param in resnet.parameters():
    #     param.requires_grad = False

    # # new final layer with 16 classes
    # num_ftrs = resnet.fc.in_features
    # resnet.fc = nn.Linear(num_ftrs, NumClasses)
    # if use_gpu:
    #     resnet = resnet.cuda()
    
    # criterion = nn.CrossEntropyLoss()

    # lr = Params['LearningRate']
    # momentum = Params['Momentum']
    # step_size = Params['StepSize']
    # gamma = Params['Gamma']

    # optimizer = optim.SGD(resnet.fc.parameters(), lr, momentum)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)

    # dataLoaders = { 'train':trainLoader, 'valid':validLoader }

    # # Training and Validate 
    # num_epochs = Params['NumEpochs']
    # start_time = time.time()
    # # model = train_model(
    # #     dataLoaders, resnet, criterion, optimizer, exp_lr_scheduler, num_epochs
    # # )
    # print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))    


if __name__=='__main__':
    main()


