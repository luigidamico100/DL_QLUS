#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:39:22 2021

@author: luigidamico
"""

import torch
from torch import nn
from torchmetrics import Accuracy, MeanAbsolutePercentageError


on_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if on_cuda else "cpu")
DATASET_PATH = '/mnt/disk0/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f' if on_cuda else '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'
num_workers = 0
MODEL_PATH = "../Experiments/exp1/model_last.pt"



''' Most common to edit'''
OUTFOLDER_PATH = '../Experiments/exp9/'
comment_text = "classification problem runned with new loss and metrics"
classification = True
batch_size = 16 if on_cuda else 32



''' Problem definition parameters'''
# Mode to choose from [random_frame_from_clip, entire_clip, entire_clip_1ch]
mode = 'random_frame_from_clip'


''' Model parameters '''
# Models to choose from [resnet (11M params), efficientnet-b0 (4M params), alexnet, vgg, squeezenet, densenet, inception]
model_name = 'efficientnet-b0'
use_pretrained = True
feature_extract = False     #Set to False to fine-tune the model.


''' Training parameters'''
fold_test = 9
num_epochs = 20 if on_cuda else 2
replicate_all_classes = 1
regularization = None
lr = 1e-4

''' classification or regression parameters'''

#for classification problem
classification_classes = ['BEST', 'RDS']
num_classes = len(classification_classes)




def get_problem_stuff():
    if classification:
        criterion = nn.CrossEntropyLoss()
        metric = Accuracy().to(device)
    else:
        criterion = nn.MSELoss()
        metric = MeanAbsolutePercentageError().to(device)
    return (criterion, metric)

info_text = "Classification: {} \nModel name: {}\n\tfold_test = {}\n{}".format(classification, model_name, fold_test, comment_text)





