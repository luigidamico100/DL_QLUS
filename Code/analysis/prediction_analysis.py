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
from dataloader_utils import train_img_transform
from dataloader_utils import NUM_ROWS
import matplotlib

np.set_printoptions(precision=4)

#%% Load dataset
RESULT_PATH = '/Volumes/SD Card/Thesis/Experiments/models_training/experiment_allfold_exp_3/'
out_file_folder = RESULT_PATH+ 'figures/'
height_size = 240

DATASET_RESULT_PATH = RESULT_PATH + 'evaluation_dataframe.csv'
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
dataset_test_wrongPrediction_florence_RDS = dataset_test_wrongPrediction_florence[dataset_test_wrongPrediction_florence['label']==1]
dataset_test_wrongPrediction_milan_healthy = dataset_test_wrongPrediction_milan[dataset_test_wrongPrediction_milan['label']==0]
dataset_test_wrongPrediction_milan_RDS = dataset_test_wrongPrediction_milan[dataset_test_wrongPrediction_milan['label']==1]
dataset_test_wrongPrediction_naples_healthy = dataset_test_wrongPrediction_naples[dataset_test_wrongPrediction_naples['label']==0]
dataset_test_wrongPrediction_naples_RDS = dataset_test_wrongPrediction_naples[dataset_test_wrongPrediction_naples['label']==1]

dataset_test_rightPrediction_BEST = dataset_test_rightPrediction[dataset_test_rightPrediction['label']==0]
dataset_test_rightPrediction_RDS = dataset_test_rightPrediction[dataset_test_rightPrediction['label']==1]
dataset_test_wrongPrediction_BEST = dataset_test_wrongPrediction[dataset_test_wrongPrediction['label']==0]
dataset_test_wrongPrediction_RDS = dataset_test_wrongPrediction[dataset_test_wrongPrediction['label']==1]


#%% Check extreme correct evaluation
n_steps = 10
'''
Healthy (Naples)
'''
sample_key = 'BEST/B_23_1_4.mat\tf0'
img = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
Healthy (Milan)
'''
sample_key = 'BEST/B_86_1_7.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
Healthy (Florence)
'''
sample_key = 'BEST/B_70_1_2.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Naples)
'''
sample_key = 'RDS/R_34_1_3.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Milan)
'''
sample_key = 'RDS/R_36_1_4.mat\tf0'
analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Florence)
'''
sample_key = 'RDS/R_29_1_5.mat\tf0'
# sample_key = 'RDS/R_47_1_2.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'Attribution_FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')


#%% Metrics for different hospitals

analysis_util.print_prediction_stats(dataset_test)

#%% Showing histories data

analysis_util.show_training_histories_byFold(hists_path=RESULT_PATH)


#%% Analysing wrong-predicted frames OLD
n_steps = 10

'''
Healthy (Naples)
'''
sample_key = 'BEST/B_11_1_3.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_NaplesHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')


'''
Healthy (Florence)
'''
# sample_key = 'BEST/B_75_1_5.mat\tf0 !ABSENTE'
# analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
# analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
# _ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_FlorenceHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')


'''
Healthy (Milan)
'''
sample_key = 'BEST/B_85_1_5.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_MilanHealthy_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Naples)
'''
sample_key = 'RDS/R_51_2_6.mat\tf0'
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_NaplesRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')

'''
RDS (Milan)
'''
sample_key = 'RDS/R_24_3_5.mat\tf0'
analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_MilanRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')


'''
RDS (Florence)
'''
sample_key = 'RDS/R_40_2_2.mat\tf0'
analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=out_file_folder+'predWrong_FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'predWrong_Attribution_FlorenceRDS_'+sample_key.replace('/','-').replace('\t','')+'.png')



''' 
Naples hospital
'''

sample_key = 'BEST/B_2_1_4.mat\tf2'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'BEST/B_11_1_2.mat\tf5'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'BEST/B_4_1_9.mat\tf4'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_51_2_4.mat\tf1'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')


sample_key = 'RDS/R_51_2_1.mat\tf5'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_33_1_6.mat\tf2'     #Naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')


'''
Paziente unpredictable
'''
sample_key = 'RDS/R_24_2_7.mat\tf2'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_24_3_6.mat\tf2'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_24_1_1.mat\tf5'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_24_2_3.mat\tf3'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_24_3_7.mat\tf4'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')


sample_key = 'RDS/R_24_2_5.mat\tf4'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_24_3_1.mat\tf0'     #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/unpredictable/'+sample_key.replace('/','-').replace('\t','')+'.png')



'''
RDS from Florence
'''

sample_key = 'RDS/R_40_2_1.mat\tf2'     #Florence RDS
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/florenceRDS/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_40_2_2.mat\tf3'     #Florence RDS
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/florenceRDS/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_47_2_4.mat\tf0'     #Florence RDS
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/florenceRDS/'+sample_key.replace('/','-').replace('\t','')+'.png')




'''
Very hard to predict
'''
sample_key = 'RDS/R_24_3_6.mat\tf0' #Milan
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'RDS/R_53_2_2.mat\tf1' #naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'BEST/B_16_1_5.mat\tf1' #naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

sample_key = 'BEST/B_22_1_8.mat\tf2' #naples
_ = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
_ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=out_file_folder+'attribution_wrongPrediction/'+sample_key.replace('/','-').replace('\t','')+'.png')

BEST/B_85_1_6.mat   #Milan
RDS/R_34_1_5.mat
RDS/R_40_2_2.mat
RDS/R_47_2_4.mat    #Florence
BEST/B_22_1_5.mat   #Naples
BEST/B_14_1_4.mat   #naples
RDS/R_41_2_4.mat    #Florence


#%% Video level
(dataset_test_videoLevel, dataset_test_videoLevel_rightPredition, dataset_test_videoLevel_wrongPrediction), acc = analysis_util.create_dataset_videoLevel(dataset_test, th=0.5)

analysis_util.print_prediction_stats(dataset_test_videoLevel)
from scipy.stats import spearmanr
spearmanr(dataset_test_videoLevel['nn_output_prob_label0'], dataset_test_videoLevel['EMOGAS index'])


#%% Augmentation tests
sample_key = 'BEST/B_70_1_2.mat\tf0'
analysis_util.show_augmentations(dataset_test, sample_key, num_aug=6, out_file_path='/Volumes/SD Card/Thesis/Experiments/augmentation/augmentation_example.jpg', train_transformation=True, old_transformation=False)
sample_key = 'RDS/R_36_1_4.mat\tf0'
analysis_util.show_augmentations(dataset_test, sample_key, num_aug=6, out_file_path='/Volumes/SD Card/Thesis/Experiments/augmentation/augmentation_example.jpg', train_transformation=True, old_transformation=False)
analysis_util.show_augmentations(dataset_test, sample_key, num_aug=6, out_file_path='/Volumes/SD Card/Thesis/Experiments/augmentation/augmentation_example.jpg', train_transformation=True, old_transformation=True)
analysis_util.show_augmentations(dataset_test, sample_key, num_aug=6, out_file_path='/Volumes/SD Card/Thesis/Experiments/augmentation/augmentation_example.jpg', train_transformation=False, old_transformation=False)

analysis_util.show_augmentations(dataset_test, sample_key, num_aug=6, out_file_path='/Volumes/SD Card/Thesis/Experiments/augmentation/augmentation_example.jpg', train_transformation=False, old_transformation=True)



#%% Folds counts

# dataset_fold = pd.DataFrame(columns=['fold', 'ospedale', 'classe', 'count'])

# for fold in range(0,10):
#     dataset_test_fold = dataset_test[dataset_test['fold']==fold]
#     df_fold = analysis_util.create_dataframe_videos(dataset_test_fold)
#     for ospedale in dataset_test['ospedale'].unique():
#         df_ospedale = df_fold[df_fold['ospedale']==ospedale]
#         for classe in dataset_test['classe'].unique():
#             df_classe = df_ospedale[df_ospedale['classe']==classe]
#             d = {'fold': fold,
#                  'ospedale': ospedale,
#                  'classe': classe,
#                  'count': len(df_classe),
#                 }
#             dataset_fold = dataset_fold.append(d, ignore_index=True)
            

import seaborn as sns
matplotlib.rcParams.update({'font.size': 18})
dataset_video_test = analysis_util.create_dataframe_video_byFold(dataset_test)
ax = sns.displot(data=dataset_video_test, x='fold', row='classe', col='ospedale', bins=10)
ax.savefig('/Volumes/SD Card/Thesis/Experiments/folds_stats/folds_stats.jpg',dpi=300)



