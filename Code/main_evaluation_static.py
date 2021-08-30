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
from config import regularization, criterion, num_epochs, OUTFOLDER_PATH, info_text, MODEL_PATH
import torch



#%% Evaluate model
if __name__ == '__main__':
    
    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = dataload_ut.get_mat_dataloaders(dataload_ut.all_classes, basePath=DATASET_PATH, num_workers=num_workers, 
                                                                                           batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                           target_value=False)
    
    model_eval = torch.load(MODEL_PATH, map_location=device)
    acc_train = stat_mod_ut.eval_model(model_eval, train_dl, num_batches=15)
    acc_val = stat_mod_ut.eval_model(model_eval, val_dl, num_batches=10)
    acc_test = stat_mod_ut.eval_model(model_eval, test_dl, num_batches=10)
    
    print("acc_train: {:2f}\nacc_val: {:2f}\nacc_test: {:2f}\n".format(acc_train, acc_val, acc_test))
