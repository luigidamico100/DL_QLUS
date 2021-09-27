#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:11:42 2021

@author: luigidamico
"""
import pandas as pd
from config import DATASET_RAW_PATH, DATASET_PATH
from dataloader_utils import NUM_ROWS

import matplotlib.pyplot as plt
from scipy.io import loadmat
import pims
import analysis_util
import pickle
import numpy as np


#%% Load dataset

DATASET_RESULT_PATH = '../../Experiments/experiment_allfold_exp_1/evaluation_dataframe_final.csv'
dataset_test = pd.read_csv(DATASET_RESULT_PATH).drop('Unnamed: 0', axis=1)
dataset_test['ospedale'] = analysis_util.create_ospedale_column(dataset_test)

(dataset_test_rightPrediction, dataset_test_wrongPrediction), accuracy_test = analysis_util.evaluate_dataset(dataset_test)


#%% Check extreme correct evaluation
'''
Very healthy patient, predicted correct
Taking sample at idx=0, EMOGAS idx = 476 (very high)
Label = 0. Predicted correct.
Very clean pleural line, but 'broken?'
Lung is black and little of horizontal artifact
'''
analysis_util.analyze_one_video_prediction(dataset_test, 0)

'''
Very healthy patient, predicted correct
Taking sample at idx=9, EMOGAS idx = 476 (very high)
Label = 0. Predicted correct.
Very clean pleural line'
Lung is black and lot of horizontal artifact
'''
analysis_util.analyze_one_video_prediction(dataset_test, 9)

'''
Very healthy patient, predicted correct
Taking sample at idx=3750, EMOGAS idx = 476 (very high)
Label = 0. Predicted correct.
Very clean pleural line but broken'
Lung is black and no white arctifact
'''
analysis_util.analyze_one_video_prediction(dataset_test, 3750)


'''
Very sick patient, predicted correct
Taking sample at idx=4015, EMOGAS idx = 101 (very low)
Label = 1. Predicted correct.
Comment on pleural line? Looks clean
Lung is full of liquid
'''
analysis_util.analyze_one_video_prediction(dataset_test, 4015)


#%%
accuracies = []
for fold in [0,1,2,3,4,5,6,7,8,9]:
    dataset_test_fold = dataset_test[dataset_test['fold']==fold]
    (_, _), acc = analysis_util.evaluate_dataset(dataset_test_fold)
    print(acc)
    accuracies = accuracies + [acc]
    hist_path = '/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_1/exp_fold_'+str(fold)+'/hist.pkl'
    hist_data = pickle.load(open(hist_path,'rb'))
    epoch_best_valLoss = np.array(hist_data[0][1][1]).argmin()
    print(hist_data[1][1][2][epoch_best_valLoss])
    print(hist_data[1][1][2][-1])



(dataset_test_videoLevel, dataset_test_videoLevel_rightPrediction, dataset_test_videoLevel_wrongPrediction), accuracy = analysis_util.create_dataset_videoLevel(dataset_test)









