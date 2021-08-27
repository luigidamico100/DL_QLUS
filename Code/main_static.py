#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:57:10 2021

@author: luigidamico
"""

from model import static_model_utils as stat_mod_ut
from dataloader import dataloader_utils as dataload_ut
import torch.optim as optim
from torch import nn
import torch

on_cuda = torch.cuda.is_available()

if on_cuda:
    DATASET_PATH = '/mnt/disk0/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f'
    num_workers = 0
else:
    DATASET_PATH = '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'
    num_workers = 0

OUT_FOLDER = '../Experiments/'


#%% Parameters definition
# Models to choose from [resnet (11M params), efficientnet-b0 (4M params), alexnet, vgg, squeezenet, densenet, inception]
# Mode to choose from [random_frame_from_clip, entire_clip, entire_clip_1ch]
model_name = 'efficientnet-b0'
use_pretrained = False
num_classes = 3
num_epochs = 2 if on_cuda else 2
batch_size = 64 if on_cuda else 2
feature_extract = False     #Set to False to fine-tune the model. 
mode = 'random_frame_from_clip'


#%% Set up model architecture

model_ft, input_size = stat_mod_ut.initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)
# print(model_ft)
# stat_mod_ut.print_model_parameters(model_ft)
params_to_update = stat_mod_ut.get_params_to_update(model_ft, feature_extract)
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update, lr=1e-4)
criterion = nn.CrossEntropyLoss()
regularization = 'l1'


#%% Dataloaders

train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = dataload_ut.get_mat_dataloaders(dataload_ut.all_classes, basePath=DATASET_PATH, num_workers=num_workers, 
                                                                                       batch_size=batch_size, mode=mode, replicate_all_classes=2,
                                                                                       target_value=False)
dataloaders_dict = {
    'train' : train_dl, 
    'val' : val_dl
    }
train_dl_it = iter(train_dl)
sample = next(train_dl_it)
val_dl_it = iter(val_dl)
test_dl_it = iter(test_dl)
print('num_iter: {:.2f}'.format(len(train_ds[0])/batch_size))


#%% Train and evaluate

models, hist = stat_mod_ut.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), regularization=regularization)

#%%


stat_mod_ut.plot_and_save(models, hist, out_folder = OUT_FOLDER)






