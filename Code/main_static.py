#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:57:10 2021

@author: luigidamico
"""

from model import static_model_utils as stat_mod_ut
from dataloader import dataloader_utils as dataload_ut
import torch.optim as optim
from config import model_name, num_classes, feature_extract, use_pretrained, classification
from config import DATASET_PATH, num_workers, batch_size, mode, replicate_all_classes, fold_test
from config import regularization, criterion, num_epochs, OUTFOLDER_PATH, info_text, classification_classes


#%% MAIN
if __name__ == '__main__':
    

#%% Set up model architecture

    model_ft, input_size = stat_mod_ut.initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)
    # print(model_ft)
    # stat_mod_ut.print_model_parameters(model_ft)
    params_to_update = stat_mod_ut.get_params_to_update(model_ft, feature_extract)
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=1e-4)

#%% Dataloaders

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = dataload_ut.get_mat_dataloaders(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                           batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                           target_value=not classification)
    dataloaders_dict = {
        'train' : train_dl, 
        'val' : val_dl,
        'test' : test_dl
        }
    train_dl_it = iter(train_dl)
    sample = next(train_dl_it)
    val_dl_it = iter(val_dl)
    test_dl_it = iter(test_dl)
    print('num_iter for training: {:.2f}'.format((len(train_ds[0]) + len(train_ds[1]))/batch_size))
    print('num_iter for val: {:.2f}'.format((len(val_ds[0]) + len(val_ds[1]))/batch_size))
    

#%% Train and evaluate
    print("Output folder: {}\t\tcheck if  it does not exist!".format(OUTFOLDER_PATH))
    models, hist = stat_mod_ut.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, 
                                           is_inception=(model_name=="inception"), regularization=regularization)

#%% 
    stat_mod_ut.plot_and_save(models, hist, out_folder = OUTFOLDER_PATH, info_text=info_text)
    

    
