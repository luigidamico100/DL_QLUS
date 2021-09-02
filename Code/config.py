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
OUTFOLDER_PATH = '../Experiments/exp9/'
MODEL_PATH = "../Experiments/exp1/model_last.pt"
comment_text = "classification problem runned with new loss and metrics"


''' Problem definition parameters'''
# Mode to choose from [random_frame_from_clip, entire_clip, entire_clip_1ch]
mode = 'random_frame_from_clip'


''' Model parameters '''
# Models to choose from [resnet (11M params), efficientnet-b0 (4M params), alexnet, vgg, squeezenet, densenet, inception]
model_name = 'efficientnet-b0'
use_pretrained = True
feature_extract = False     #Set to False to fine-tune the model.


''' Training parameters'''
fold_test = 0
num_epochs = 20 if on_cuda else 2
batch_size = 16 if on_cuda else 32
replicate_all_classes = 1
regularization = None
lr = 1e-4

''' classification or regression parameters'''
classification = False

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

info_text = "Classification: {} \n\tfold_test = {}\n{}".format(classification, fold_test, comment_text)

# outputs = torch.rand(32,2).type(torch.FloatTensor)
# labels = torch.rand(32,1).type(torch.LongTensor)
# print(criterion(outputs, labels))
# print(metric(outputs, labels))



# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
# x, y = Variable(x), Variable(y)




# #%% trying torchmetrics

# import torch
# from torchmetrics import Accuracy, MeanAbsolutePercentageError
# target = torch.tensor([0, 1, 2, 3]).unsqueeze(dim=1)
# preds = torch.tensor([0, 2, 1, 3]).unsqueeze(dim=1)
# accuracy = Accuracy()
# print(accuracy(preds, target))


# from torchmetrics import MeanAbsolutePercentageError
# target = torch.tensor([1, 10, 1e6]).unsqueeze(dim=1)
# preds = torch.tensor([0.9, 15, 1.2e6]).unsqueeze(dim=1)
# mean_abs_percentage_error = MeanAbsolutePercentageError()
# mean_abs_percentage_error(preds, target)








