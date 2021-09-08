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
DATASET_PATH = '/mnt/disk2/diego.gragnaniello/Eco/ICPR/Dataset_processato/Dataset_f' if on_cuda else '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'
num_workers = 0


experiment_all_fold = False
''' Most common for training'''
OUTFOLDER_PATH = '../Experiments/exp_9/'     # used in case experiment_all_fold=False
OUTFOLDER_ALLFOLD_FOLDER = '../Experiments/experiment_allfold_exp_1/'    # used in case experiment_all_fold=True
classification = True
batch_size = 4 if on_cuda else 4
comment_text = "...."

''' Model evaluation '''
MODEL_PATH = '../Experiments/experiment_allfold_exp_1/exp_fold_0/model_best.pt'
ALLFOLD_MODELS_FOLDER = '../Experiments/experiment_allfold_exp_0/'


''' Problem definition parameters'''
# Mode to choose from [random_frame_from_clip_old, random_frame_from_clip, fixed_number_of_frames, fixed_number_of_frames_1ch]
mode = 'random_frame_from_clip'


''' Model parameters '''
# Models to choose from [resnet (11M params), efficientnet-b0 (4M params), alexnet, vgg, squeezenet, densenet, inception]
model_name = 'efficientnet-b0'
use_pretrained = True
feature_extract = False     #Set to False to fine-tune the model.


''' Training parameters'''
fold_test_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fold_test = 9
num_epochs = 20 if on_cuda else 2
replicate_all_classes = 1
regularization = None
lr = 1e-4

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


info_text = "Model name: {}\nClassification = {}\n\tfold_test = {}\nBatch size = {}\n{}".format(model_name, classification, fold_test, batch_size, comment_text)
