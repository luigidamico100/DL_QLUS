#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 00:09:07 2021

@author: luigidamico
"""

import analysis_util
import pandas as pd


#%%
# analysis_util.create_and_save_dataframe_train(out_path='../../Experiments/train_dataset_target_value.csv', target_value=True)
dataframe_train = pd.read_csv('/Volumes/SD Card/Thesis/Experiments/train_dataset.csv').drop('Unnamed: 0', axis=1)
dataframe_videos = analysis_util.create_and_save_dataframe_videos(dataframe_train)
dataframe_patients = analysis_util.create_and_save_dataframe_patients(dataframe_videos)

#%% Sample counting for different hospitals and classes
print(dataframe_videos['ospedale'].value_counts())
print(dataframe_videos['classe'].value_counts())

analysis_util.print_dataframe_stats(dataframe_videos)


print(dataframe_patients['ospedale'].value_counts())
print(dataframe_patients['classe'].value_counts())
analysis_util.print_dataframe_stats(dataframe_patients)




