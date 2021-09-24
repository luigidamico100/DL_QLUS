#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 00:09:07 2021

@author: luigidamico
"""

import analysis_util as an_ut
import pandas as pd


#%%

dataframe_train = pd.read_csv('../../Experiments/train_dataset.csv').drop('Unnamed: 0', axis=1)
dataframe_complete = an_ut.create_and_save_dataframe_complete(dataframe_train)
dataframe_complete['ospedale'] = an_ut.create_ospedale_column(dataframe_complete)
    
dataframe_complete['ospedale'].value_counts()
dataframe_complete['classe'].value_counts()
