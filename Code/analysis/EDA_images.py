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
Healthy (Naples)
'''
sample_key = 'BEST/B_23_1_4.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_7_1_6.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_5_1_3.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
RDS (Naples)
'''
sample_key = 'RDS/R_34_1_3.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)


'''
RDS (Milan)
'''
sample_key = 'RDS/R_36_1_4.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)

'''
RDS (Florence)
'''
sample_key = 'RDS/R_29_1_5.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/figures/'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key)




#%%
(dataset_test_videoLevel, dataset_test_videoLevel_rightPrediction, dataset_test_videoLevel_wrongPrediction), accuracy = analysis_util.create_dataset_videoLevel(dataset_test)









