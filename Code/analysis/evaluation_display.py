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
from sklearn.metrics import confusion_matrix, accuracy_score


#%% Load dataset

DATASET_RESULT_PATH = '../../Experiments/experiment_allfold_exp_5/evaluation_dataframe_final.csv'
dataset_test = pd.read_csv(DATASET_RESULT_PATH, index_col='keys')
dataset_test['ospedale'] = analysis_util.create_ospedale_column(dataset_test)



(dataset_test_rightPrediction, dataset_test_wrongPrediction), accuracy_test = analysis_util.evaluate_dataset(dataset_test)

dataset_test_rightPrediction_naples = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Naples']
dataset_test_wrongPrediction_naples = dataset_test_wrongPrediction[dataset_test_wrongPrediction['ospedale']=='Naples']
dataset_test_rightPrediction_milan = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Milan']
dataset_test_wrongPrediction_milan = dataset_test_wrongPrediction[dataset_test_wrongPrediction['ospedale']=='Milan']
dataset_test_rightPrediction_florence = dataset_test_rightPrediction[dataset_test_rightPrediction['ospedale']=='Florence']
dataset_test_wrongPrediction_florence = dataset_test_wrongPrediction[dataset_test_wrongPrediction['ospedale']=='Florence']

dataset_test_wrongPrediction_florence_healthy = dataset_test_wrongPrediction_florence[dataset_test_wrongPrediction_florence['label']==0]
dataset_test_wrongPrediction_milan_healthy = dataset_test_wrongPrediction_milan[dataset_test_wrongPrediction_milan['label']==0]
dataset_test_wrongPrediction_milan_RDS = dataset_test_wrongPrediction_milan[dataset_test_wrongPrediction_milan['label']==1]


#%% Check extreme correct evaluation
out_file_folder = '/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_5/figures/'

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_23_1_4.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
Healthy (Milan)
'''
sample_key = 'BEST/B_86_1_7.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
Healthy (Florence)
'''
sample_key = 'BEST/B_70_1_2.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Naples)
'''
sample_key = 'RDS/R_34_1_3.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')


'''
RDS (Milan)
'''
sample_key = 'RDS/R_36_1_4.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Florence)
'''
sample_key = 'RDS/R_29_1_5.mat\tf0'
analysis_util.save_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=out_file_folder+'Attribution_FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')


#%% Metrics for different hospitals
print(accuracy_score(dataset_test['label'], dataset_test['label_prediction']))
print(confusion_matrix(dataset_test['label'], dataset_test['label_prediction']))
print('------------------Naples') 
dataset_test_naples = dataset_test[dataset_test['ospedale']=='Naples']
print(accuracy_score(dataset_test_naples['label'], dataset_test_naples['label_prediction']))
print(confusion_matrix(dataset_test_naples['label'], dataset_test_naples['label_prediction'], normalize='true'))
print(confusion_matrix(dataset_test_naples['label'], dataset_test_naples['label_prediction']))
print('------------------Florence')
dataset_test_florence = dataset_test[dataset_test['ospedale']=='Florence']
print(accuracy_score(dataset_test_florence['label'], dataset_test_florence['label_prediction']))
print(confusion_matrix(dataset_test_florence['label'], dataset_test_florence['label_prediction'], normalize='true'))
print(confusion_matrix(dataset_test_florence['label'], dataset_test_florence['label_prediction']))
print('------------------Milan')
dataset_test_milan = dataset_test[dataset_test['ospedale']=='Milan']
print(accuracy_score(dataset_test_milan['label'], dataset_test_milan['label_prediction']))
print(confusion_matrix(dataset_test_milan['label'], dataset_test_milan['label_prediction'], normalize='true'))
print(confusion_matrix(dataset_test_milan['label'], dataset_test_milan['label_prediction']))

#%% Showing histories data

analysis_util.show_training_histories_byFold(hists_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_5/')

#%% Analysing wrong-predicted frames

dataset_test_wrongPrediction_naples_healthy

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_11_1_1.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=None)

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_21_1_1.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=None)

'''
Healthy (Florence)
'''
sample_key = 'BEST/B_75_1_5.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=None)


'''
Healthy (Milan)
'''
sample_key = 'BEST/B_84_1_5.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=None)

'''
RDS (Milan)
'''
sample_key = 'RDS/R_24_3_5.mat\tf0'
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=20, out_file_path=None)




#%% Video level
_, acc = analysis_util.create_dataset_videoLevel(dataset_test, th=0.5)

