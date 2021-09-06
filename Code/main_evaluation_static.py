#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:10:10 2021

@author: luigidamico
"""

from dataloader import dataloader_utils as dataload_ut
from model import static_model_utils as stat_mod_ut
from config import model_name, num_classes, feature_extract, use_pretrained, device
from config import DATASET_PATH, num_workers, batch_size, mode, replicate_all_classes
from config import regularization, num_epochs, OUTFOLDER_PATH, info_text, MODEL_PATH
from config import classification, fold_test, classification_classes
import config
import torch
from torch import nn



#%% Evaluate model
if __name__ == '__main__':
    

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = dataload_ut.get_mat_dataloaders(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                               batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                               target_value=not classification)

    
    loss, metric = config.get_problem_stuff()
    model_eval = torch.load(MODEL_PATH, map_location=device)
    
    #%%
    
    train_score, train_metric, train_auc_score = stat_mod_ut.eval_model(model_eval, train_dl, loss, metric, num_batches=5)
    val_score, val_metric = stat_mod_ut.eval_model(model_eval, val_dl, loss, metric, num_batches=5)
    test_score, test_metric = stat_mod_ut.eval_model(model_eval, test_dl, loss, metric, num_batches=50)



