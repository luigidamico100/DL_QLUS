#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:11:42 2021

@author: luigidamico
"""
import pandas as pd
from config import DATASET_RAW_PATH, DATASET_PATH
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pims
import analysis_util
import pickle
import numpy as np


#%% Load dataset

DATASET_RESULT_PATH = '../../Experiments/experiment_allfold_exp_3/evaluation_dataframe_final.csv'
dataset_test = pd.read_csv(DATASET_RESULT_PATH, index_col='keys')
dataset_test['ospedale'] = analysis_util.create_ospedale_column(dataset_test)

(dataset_test_rightPrediction, dataset_test_wrongPrediction), accuracy_test = analysis_util.evaluate_dataset(dataset_test)

dataset_test_rightPrediction_milan = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Milan']
dataset_test_rightPrediction_florence = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Florence']
dataset_test_rightPrediction_naples = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Naples']
dataset_test_rightPrediction_milan_healthy = dataset_test_rightPrediction_milan[dataset_test_rightPrediction_milan['label']==0]
dataset_test_rightPrediction_milan_sick = dataset_test_rightPrediction_milan[dataset_test_rightPrediction_milan['label']==1]
dataset_test_rightPrediction_naples_healthy = dataset_test_rightPrediction_naples[dataset_test_rightPrediction_naples['label']==0]
dataset_test_rightPrediction_naples_sick = dataset_test_rightPrediction_naples[dataset_test_rightPrediction_naples['label']==1]
dataset_test_rightPrediction_florence_healthy = dataset_test_rightPrediction_florence[dataset_test_rightPrediction_florence['label']==0]
dataset_test_rightPrediction_florence_sick = dataset_test_rightPrediction_florence[dataset_test_rightPrediction_florence['label']==1]


#%% Check extreme correct evaluation

'''
Very healthy patient, predicted correct. Naples
Taking sample at idx=BEST/B_10_1_3.mat\tf0, EMOGAS idx = 476 (very high)
Label = 0. Predicted NOT TOTALLY correct.
Segmented pleural line. Horizontal arctifact. Black under the line.
'''
sample_key = 'BEST/B_10_1_3.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
Very sick patient, predicted correct. Naples
Taking sample at idx=RDS/R_34_1_3.mat\tf0, EMOGAS idx = 
Label = 1. Predicted correct.
Broken pleural line. Full of water. 
'''
sample_key = 'RDS/R_34_1_3.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
Very sick patient, predicted correct. Milan
Taking sample at idx=RDS/R_34_1_3.mat\tf0, EMOGAS idx = 
Label = 1. Predicted correct.
Lot of vertical artififact
'''
sample_key = 'RDS/R_36_1_4.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
Very sick patient, predicted correct. Florence
Taking sample at idx=RDS/R_29_1_5.mat\tf0, EMOGAS idx = 
Label = 1. Predicted correct.
Lot of water
'''
sample_key = 'RDS/R_29_1_5.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)


#%% Check accuracies for each fold

accuracies = []
for fold in [0,1,2,3,4,5,6,7,8,9]:
    dataset_test_fold = dataset_test[dataset_test['fold']==fold]
    (_, _), acc = analysis_util.evaluate_dataset(dataset_test_fold)
    print('computed acc: ', acc)
    accuracies = accuracies + [acc]
    hist_path = '/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/exp_fold_'+str(fold)+'/hist.pkl'
    hist_data = pickle.load(open(hist_path,'rb'))
    epoch_best_valLoss = np.array(hist_data[0][1][1]).argmin()
    print('hist best acc: ', hist_data[1][1][2][epoch_best_valLoss])
    print('hist last acc: ', hist_data[1][1][2][-1])
    print()


#%%
(dataset_test_videoLevel, dataset_test_videoLevel_rightPrediction, dataset_test_videoLevel_wrongPrediction), accuracy = analysis_util.create_dataset_videoLevel(dataset_test)









