#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:11:42 2021

@author: luigidamico
"""
import pandas as pd
from config import DATASET_RAW_PATH, DATASET_PATH
from dataloader.dataloader_utils import NUM_ROWS

import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat

#%%

DATASET_RESULT_PATH = '../Experiments/experiment_allfold_exp_1/evaluation_dataframe_final.csv'
dataset = pd.read_csv(DATASET_RESULT_PATH)

dataset_wrong_prediction = dataset[dataset['label_prediction'] != dataset['label']]
dataset_right_prediction = dataset[dataset['label_prediction'] == dataset['label']]

sample = dataset_wrong_prediction.iloc[300]

if sample['classe'] == 'BEST':
    clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' + sample['video_name']
else:
    clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' +  sample['esame_name'] + '/' + sample['video_name']
clip_mat_path = DATASET_PATH + '/' + sample['processed_video_name']


#%%

import pims
clip_raw = pims.Video(clip_raw_path)
img_raw = clip_raw[-1]
plt.imshow(img_raw), plt.show()

matdata = loadmat(clip_mat_path)
clip_mat = [matdata[k] for k in matdata.keys() if k.startswith('f') and len(k) < 3]
for img_mat in clip_mat:
    plt.imshow(img_mat), plt.show()


#%%
import sklearn
sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)











