#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:10:10 2021

@author: luigidamico
"""

from dataloader import dataloader_utils as dataload_ut
from model import static_model_utils as stat_mod_ut
from config import model_name, num_classes, feature_extract, use_pretrained, device, fold_test_list
from config import DATASET_PATH, num_workers, batch_size, mode, replicate_all_classes
from config import regularization, num_epochs, ALLFOLD_MODELS_FOLDER, info_text, MODEL_PATH
from config import classification, fold_test, classification_classes, experiment_all_fold, on_cuda
import config
import torch
from torch import nn
import numpy as np


#%% Evaluate model
if __name__ == '__main__':
    num_batches = 10e10 if on_cuda else 2
    
    if not experiment_all_fold:
        print('##################### Evaluating model in: ' + MODEL_PATH + ' #####################\n')
        model_evaluation = torch.load(MODEL_PATH, map_location=device)        

        dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                   batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                   target_value=not classification)
        loss, metric = config.get_problem_stuff()
        
        _, train_score, train_metric = stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['train'], loss, metric, num_batches=2)
        _, val_score, val_metric= stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['val'], loss, metric, num_batches=2)
        _, test_score, test_metric= stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['test'], loss, metric, num_batches=2)
        
        print('TRAIN evaluation -- \n\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), train_score, str(metric), train_metric))
        print('VAL evaluation -- \n\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), val_score, str(metric), val_metric))
        print('TEST evaluation -- \n\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), test_score, str(metric), test_metric))
        
        if classification:
            dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                       batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                       target_value=True)
            train_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['train'], num_batches=2)
            val_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['val'], num_batches=2)
            test_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['test'], num_batches=2)
            print('TRAIN evaluation -- \n\t\tSpearman Corr: {:.2f}'.format(train_spearmanCorr))
            print('VAL evaluation -- \n\t\tSpearman Corr: {:.2f}'.format(val_spearmanCorr))
            print('TEST evaluation -- \n\t\tSpearman Corr: {:.2f}'.format(test_spearmanCorr))
        
            
    else:
        print('##################### Evaluating all models in: ' + ALLFOLD_MODELS_FOLDER + ' #####################\n')
        running_metrics = np.empty((0,2))
        for fold_test in fold_test_list:
            print('##################### Evaluating new fold: ' + str(fold_test) + ' #####################')
            dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                            batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                            target_value=True)

            loss, metric = config.get_problem_stuff()
            inputModel_path = ALLFOLD_MODELS_FOLDER + 'exp_fold_{}/'.format(fold_test) + 'model_best.pt'
            model_evaluation = torch.load(inputModel_path, map_location=device)
            n_samples, _, test_metric= stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['test'], loss, metric, num_batches=num_batches)
            running_metrics = np.concatenate((running_metrics, np.array([[n_samples, test_metric]])))
        
        n_tot_samples = running_metrics.sum(axis=0)[0]
        metrics_weights = running_metrics[:,0] / n_tot_samples
        metric_weighted = (running_metrics[:,1] * metrics_weights).sum()
        print('------------------- Results------------------- ')
        print('Test ' + str(metric)+ ' for each fold: ')
        print(running_metrics)
        print('\nAggregate ' + str(metric) + ': '+str(metric_weighted))


