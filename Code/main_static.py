#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:57:10 2021

@author: luigidamico
"""

import static_model_utils as stat_mod_ut
import dataloader_utils as dataload_ut
import torch.optim as optim
from config import model_name, num_classes, feature_extract, use_pretrained, classification
from config import DATASET_PATH, num_workers, batch_size, mode, replicate_all_classes, fold_test
from config import regularization, num_epochs, OUTFOLDER_PATH, info_text
from config import classification_classes, lr, experiment_all_fold, fold_test_list, OUTFOLDER_ALLFOLD_FOLDER
import config
import os


#%% MAIN
if __name__ == '__main__':

#%% Set up model architecture

    model_ft, height_size = stat_mod_ut.initialize_model(model_name, classification, num_classes, feature_extract, use_pretrained=use_pretrained)
    print(model_ft)
    stat_mod_ut.print_model_parameters(model_ft)
    params_to_update = stat_mod_ut.get_params_to_update(model_ft, feature_extract)
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=lr)
    criterion, metric = config.get_problem_stuff()
    
    if not experiment_all_fold:
        dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test, height_size=height_size,
                                                                                       batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                       target_value=not classification)
        
        print("Output folder: {}\t\tBe sure that it does not exist!".format(OUTFOLDER_PATH))
        models, hist = stat_mod_ut.train_model(model_ft, dataloaders_dict, criterion, metric, optimizer_ft, num_epochs=num_epochs, 
                                               is_inception=(model_name=="inception"), regularization=regularization)

        stat_mod_ut.plot_and_save(models, hist, out_folder=OUTFOLDER_PATH, info_text=info_text)
    
    else:
        print("Output folder: {}".format(OUTFOLDER_ALLFOLD_FOLDER))
        os.makedirs(OUTFOLDER_ALLFOLD_FOLDER, exist_ok=True)
        for fold_test in fold_test_list:
            dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test, height_size=height_size,
                                                                                           batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                           target_value=not classification)            
    
            outfolder_path = OUTFOLDER_ALLFOLD_FOLDER + 'exp_fold_{}/'.format(fold_test)
            print("Outout fold folder: {}\t\tBe sure that it does not exist".format(outfolder_path))
            models, hist = stat_mod_ut.train_model(model_ft, dataloaders_dict, criterion, metric, optimizer_ft, num_epochs=num_epochs, 
                                                   is_inception=(model_name=="inception"), regularization=regularization)
    
            stat_mod_ut.plot_and_save(models, hist, out_folder=outfolder_path, info_text=info_text)
        


