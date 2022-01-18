#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:51:29 2021

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

#%%

RESULT_PATH = '/Volumes/SD Card/Thesis/Experiments/models_training/experiment_allfold_exp_3/'
out_file_folder = RESULT_PATH+ 'figures/'
height_size = 240

DATASET_RESULT_PATH = RESULT_PATH + 'evaluation_dataframe.csv'
dataset_test = pd.read_csv(DATASET_RESULT_PATH, index_col='keys')
dataset_test['ospedale'] = analysis_util.create_ospedale_column(dataset_test)

(dataset_test_rightPrediction, dataset_test_wrongPrediction), accuracy_test = analysis_util.evaluate_dataset(dataset_test)

#%% Show wrong predictions
    
df = dataset_test_wrongPrediction_florence
feature = 'bimbo_name'
feature = 'processed_video_name'
for label in df[feature].unique():
    print(label)
    df_ = df[dataset_test_wrongPrediction[feature]==label]
    idx = df_.index[0]
    print('video_name: ' + df.loc[idx]['processed_video_name'])
    print('hospital: ' + df.loc[idx]['ospedale'])
    _ = analysis_util.get_video_frame(df, idx, out_file_path=None)
    # analysis_util.analyze_one_video_prediction(dataset_test, idx)
    input('...press enter...')


#%% Check some wrong predictions
n_steps = 10

def checking_function(dataset_test, sample_key):
    print(dataset_test.loc[sample_key]['ospedale'])
    
    img = analysis_util.get_video_frame(dataset_test, sample_key, out_file_path=None)
    analysis_util.analyze_one_video_prediction(dataset_test, sample_key)
    _ = analysis_util.show_sample_attribution(dataset_test, sample_key, n_steps=n_steps, height_size=height_size, show_original_img=True, input_model_path=RESULT_PATH, out_file_path=None)


checking_function(dataset_test, 'BEST/B_70_1_2.mat\tf0')

'''
Possiede sia a-lines che b-lines.
La rete si concentra sui pixels che stanno in basso... 
Provengono da Napoli
'''
checking_function(dataset_test, 'RDS/R_51_2_4.mat\tf0')
checking_function(dataset_test, 'RDS/R_33_1_6.mat\tf0')
checking_function(dataset_test, 'RDS/R_33_1_5.mat\tf0')
checking_function(dataset_test, 'BEST/B_4_1_1.mat\tf0')
checking_function(dataset_test, 'BEST/B_6_1_1.mat\tf0')


'''
Too saturated
L’immagine mostra chiaramente presenza di liquido. 
La rete si concentra poco sui pixel in basso alla linea pleurica.
'''
checking_function(dataset_test, 'RDS/R_24_2_7.mat\tf5')
BEST/B_85_1_6.mat


'''
La rete si concentra sia sulle a-lines che sui pixels in basso
'''
checking_function(dataset_test, 'BEST/B_2_1_4.mat\tf2')     #Naples
checking_function(dataset_test, 'BEST/B_15_1_2.mat\tf5')    #Naples
checking_function(dataset_test, 'BEST/B_20_1_8.mat\tf3')


'''
Difficile da riconoscere!
'''
checking_function(dataset_test, 'RDS/R_24_3_6.mat\tf0')     #Milan
checking_function(dataset_test, 'BEST/B_85_1_5.mat\tf0')    #Milan. Solo un frame è difficile da predirre. Il resto predice bene
checking_function(dataset_test, 'RDS/R_53_2_2.mat\tf0')     #Naples
checking_function(dataset_test, 'BEST/B_16_1_5.mat\tf1')    #Naples
checking_function(dataset_test, 'BEST/B_14_1_4.mat\tf3')
checking_function(dataset_test, 'RDS/R_41_2_4.mat\tf0')     # Florence. Solo un frame è difficiel da predirre. Il resto predice bene
checking_function(dataset_test, 'BEST/B_6_1_1.mat\tf5')
checking_function(dataset_test, 'BEST/B_22_1_5.mat\tf0')    #Solo un frame viene predetto male
checking_function(dataset_test, 'BEST/B_14_1_4.mat\tf0')    # solo due frame sono prededdi male
checking_function(dataset_test, 'BEST/B_22_1_8.mat\tf2')



'''
La rete si concentra sui pixel superiori alle linea pleurica.
'''
checking_function(dataset_test, 'RDS/R_51_2_1.mat\tf5')
checking_function(dataset_test, 'BEST/B_22_1_5.mat\tf5')
checking_function(dataset_test, 'BEST/B_11_1_2.mat\tf5')


'''
Si concetra poi sulle parti oscurate tra le b-lines.
'''
checking_function(dataset_test, 'RDS/R_51_2_5.mat\tf0')
checking_function(dataset_test, 'RDS/R_47_2_4.mat\tf0')


'''
La rete non riesce a riconoscere la linea pleurica
'''
checking_function(dataset_test, 'RDS/R_24_2_6.mat\tf2')     #Milan


'''
to choose
'''
checking_function(dataset_test, 'BEST/B_21_1_1.mat\tf0')
checking_function(dataset_test, 'RDS/R_34_2_4.mat\tf0')
checking_function(dataset_test, 'RDS/R_40_2_1.mat\tf0')



checking_function(dataset_test, 'BEST/B_20_1_8.mat\tf3')








BEST/B_20_1_8.mat




