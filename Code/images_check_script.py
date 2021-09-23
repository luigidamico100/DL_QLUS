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
from scipy.io import loadmat
import pims


def create_ospedale_column(dataset):
    ospedale_col = pd.Series(index=dataset.index, dtype=object)
    for idx in dataset.index:
        bimbo_name = dataset.loc[idx]['bimbo_name']
        if 'Buzzi' in bimbo_name:
            ospedale_col.loc[idx] = 'Milan'
        elif 'Firenze' in bimbo_name:
            ospedale_col.loc[idx] = 'Florence'
        else:
            ospedale_col.loc[idx] = 'Naples'
    return ospedale_col
    

def evaluate_dataset(dataset):
    dataset_wrong_prediction = dataset[dataset['label_prediction'] != dataset['label']]
    dataset_right_prediction = dataset[dataset['label_prediction'] == dataset['label']]
    accuracy = len(dataset_right_prediction) / len(dataset)
    print('Accuracy: {:.2f}'.format(accuracy))
    return (dataset_right_prediction, dataset_wrong_prediction), accuracy


def create_dataset_videoLevel(dataset):
    dataset_videoLevel = dataset.groupby(['video_name','ospedale','bimbo_name'], as_index=False).mean().set_index('video_name')
    dataset_videoLevel['label_prediction_videoLevel'] = (dataset_videoLevel['nn_output_prob_label1'] > 0.5).astype('float')
    dataset_videoLevel_wrong_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction_videoLevel'] != dataset_videoLevel['label']]
    dataset_videoLevel_right_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction_videoLevel'] == dataset_videoLevel['label']]
    accuracy = len(dataset_videoLevel_right_prediction) / len(dataset_videoLevel)
    print('Accuracy videoLevel: {:.2f}'.format(accuracy))
    return (dataset_videoLevel, dataset_videoLevel_right_prediction, dataset_videoLevel_wrong_prediction), accuracy


def analyze_one_video_prediction(dataset, idx):
    dataset_wrong_prediction = dataset[dataset['label_prediction'] != dataset['label']]
    dataset_right_prediction = dataset[dataset['label_prediction'] == dataset['label']]
    sample = dataset.loc[idx]
    video_name = sample['video_name']
    print("Taking video: "+video_name)
    wrong_prediction = dataset_wrong_prediction[dataset_wrong_prediction['video_name'] == video_name]
    right_prediction = dataset_right_prediction[dataset_right_prediction['video_name'] == video_name]
    n_frame_wrong = len(wrong_prediction)
    n_frame_right = len(right_prediction) 
    print('{} frames predicted wrong out of {} total'.format(n_frame_wrong, n_frame_wrong+n_frame_right))
    
    ### Load raw img   
    if sample['classe'] == 'BEST':
        clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' + sample['video_name']
    else:
        clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' +  sample['esame_name'] + '/' + sample['video_name']
    clip_raw = pims.Video(clip_raw_path)
    img_raw = clip_raw[-1]
    
    ### load mat imgs
    clip_mat_path = DATASET_PATH + '/' + sample['processed_video_name']
    matdata = loadmat(clip_mat_path)
    
    ### images showing
    plt.imshow(img_raw), plt.show()
    for idx in wrong_prediction.index:
        k = wrong_prediction.loc[idx]['frame_key']
        prob_label0 = wrong_prediction.loc[idx]['nn_output_prob_label0']
        correct_label = wrong_prediction.loc[idx]['classe']
        plt.imshow(matdata[k][:NUM_ROWS]), plt.title('{}, wrong prediction, correct: {}, prob_label0: {:.2f}'.format(k, correct_label, prob_label0)), plt.show()
    
    for idx in right_prediction.index:
        k = right_prediction.loc[idx]['frame_key']
        prob_label0 = right_prediction.loc[idx]['nn_output_prob_label0']
        correct_label = right_prediction.loc[idx]['classe']
        plt.imshow(matdata[k][:NUM_ROWS]), plt.title('{}, correct prediction, correct: {}, prob_label0: {:.2f}'.format(k, correct_label, prob_label0)), plt.show()
    

#%% Load dataset and split in wrong and right prediction

DATASET_RESULT_PATH = '../Experiments/experiment_allfold_exp_1/evaluation_dataframe_final.csv'
dataset = pd.read_csv(DATASET_RESULT_PATH).drop('Unnamed: 0', axis=1)
dataset['ospedale'] = create_ospedale_column(dataset)
dataset_fold0 = dataset[dataset['fold']==0.]

(dataset_right_prediction, dataset_wrong_prediction), accuracy = evaluate_dataset(dataset)

analyze_one_video_prediction(dataset, 27)

(_, _), _ = evaluate_dataset(dataset_fold0)

(dataset_videoLevel, dataset_videoLevel_right_prediction, dataset_videoLevel_wrong_prediction), accuracy = create_dataset_videoLevel(dataset)









