#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 00:09:07 2021

@author: luigidamico
"""

import analysis_util
import pandas as pd


#%%
analysis_util.create_and_save_dataframe_train(out_path='../../Experiments/train_dataset_target_value.csv', target_value=True)
dataframe_train = pd.read_csv('../../Experiments/train_dataset.csv').drop('Unnamed: 0', axis=1)
dataframe_videos = analysis_util.create_and_save_dataframe_videos(dataframe_train)
dataframe_patients = analysis_util.create_and_save_dataframe_patients(dataframe_videos)
  

#%% Sample counting for different hospitals and classes
print(dataframe_videos['ospedale'].value_counts())
print(dataframe_videos['classe'].value_counts())

for ospedale in dataframe_videos['ospedale'].unique():
    df_ospedale = dataframe_videos[dataframe_videos['ospedale']==ospedale]
    for classe in dataframe_videos['classe'].unique():
        df = df_ospedale[df_ospedale['classe']==classe]
        print('Ospedale: '+ospedale+'\tClasse: ',classe, '\ttot samples: ', len(df))
        
print(dataframe_patients['ospedale'].value_counts())
print(dataframe_patients['classe'].value_counts())
for ospedale in dataframe_patients['ospedale'].unique():
    df_ospedale = dataframe_patients[dataframe_patients['ospedale']==ospedale]
    for classe in dataframe_patients['classe'].unique():
        df = df_ospedale[df_ospedale['classe']==classe]
        print('Ospedale: '+ospedale+'\tClasse: ',classe, '\ttot samples: ', len(df)) 



