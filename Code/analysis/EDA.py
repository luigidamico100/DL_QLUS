#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 00:09:07 2021

@author: luigidamico
"""

import analysis_util
import pandas as pd


#%%

dataframe_train = pd.read_csv('../../Experiments/train_dataset.csv').drop('Unnamed: 0', axis=1)
dataframe_complete = analysis_util.create_and_save_dataframe_complete(dataframe_train)
dataframe_complete['ospedale'] = analysis_util.create_ospedale_column(dataframe_complete)
    

#%% Sample counting for different hospitals and classes
print(dataframe_complete['ospedale'].value_counts())
print(dataframe_complete['classe'].value_counts())

for ospedale in dataframe_complete['ospedale'].unique():
    df_ospedale = dataframe_complete[dataframe_complete['ospedale']==ospedale]
    for classe in dataframe_complete['classe'].unique():
        df = df_ospedale[df_ospedale['classe']==classe]
        print('Ospedale: '+ospedale+'\tClasse: ',classe, '\ttot samples: ', len(df))






