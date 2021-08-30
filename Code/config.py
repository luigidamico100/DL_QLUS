#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:39:22 2021

@author: luigidamico
"""

import torch
from torch import nn

on_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if on_cuda else "cpu")
DATASET_PATH = '/mnt/disk0/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f' if on_cuda else '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'
num_workers = 0
OUTFOLDER_PATH = '../Experiments/exp1/'
MODEL_PATH = "../Experiments/Experiments_1/model_last.pt"


''' Problem definition parameters'''
# Mode to choose from [random_frame_from_clip, entire_clip, entire_clip_1ch]
mode = 'random_frame_from_clip'
classification = True
classification_classes = ['BEST', 'RDS']
num_classes = len(classification_classes)


''' Model parameters '''
# Models to choose from [resnet (11M params), efficientnet-b0 (4M params), alexnet, vgg, squeezenet, densenet, inception]
model_name = 'efficientnet-b0'
use_pretrained = True
feature_extract = False     #Set to False to fine-tune the model.


''' Training parameters'''
fold_test = 0
num_epochs = 20 if on_cuda else 2
batch_size = 64 if on_cuda else 2
replicate_all_classes = 1
criterion = nn.CrossEntropyLoss()
regularization = None


info_text = "Binary classification \n\tfold_test = {}".format(fold_test)