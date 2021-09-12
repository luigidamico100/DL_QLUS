#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:31:40 2021

@author: luigidamico
"""

from dataloader import dataloader_utils as dataload_ut
from model import static_model_utils as stat_mod_ut
from config import model_name, num_classes, feature_extract, use_pretrained, device, fold_test_list
from config import DATASET_PATH, num_workers, batch_size, mode, replicate_all_classes, debug
from config import regularization, num_epochs, ALLFOLD_MODELS_FOLDER, info_text, MODEL_PATH
from config import classification, fold_test, classification_classes, experiment_all_fold, on_cuda
import config
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import spearmanr
from pandas import DataFrame


#%% Evaluate model
if __name__ == '__main__':
    loss, metric = config.get_problem_stuff()
    
    if not experiment_all_fold:
        print('##################### Evaluating model in: ' + MODEL_PATH + ' #####################\n')
        model_evaluation = torch.load(MODEL_PATH, map_location=device)        

        dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                   batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                   target_value=not classification)

        _, train_score, train_metric = stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['train'], loss, metric, debug=debug)
        _, val_score, val_metric= stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['val'], loss, metric, debug=debug)
        _, test_score, test_metric= stat_mod_ut.eval_model(model_evaluation, dataloaders_dict['test'], loss, metric, debug=debug)
        
        print('TRAIN evaluation\t--\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), train_score, str(metric), train_metric))
        print('VAL evaluation\t--\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), val_score, str(metric), val_metric))
        print('TEST evaluation\t--\t\t{}: {:.2f}\t{}: {:.2f}'.format(str(loss), test_score, str(metric), test_metric))
        
        if classification:
            dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                       batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                       target_value=True)
            _, train_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['train'], debug=debug)
            _, val_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['val'], debug=debug)
            _, test_spearmanCorr = stat_mod_ut.eval_spearmanCorr(model_evaluation, dataloaders_dict['test'], debug=debug)
            print('TRAIN evaluation\t--\t\tSpearman Corr: {:.2f}'.format(train_spearmanCorr))
            print('VAL evaluation\t--\t\tSpearman Corr: {:.2f}'.format(val_spearmanCorr))
            print('TEST evaluation\t--\t\tSpearman Corr: {:.2f}'.format(test_spearmanCorr))
        
            
    else:
        print('##################### Evaluating all models in: ' + ALLFOLD_MODELS_FOLDER + ' #####################\n')
        running_outputs = np.empty((0,2))
        running_outputs_prob = np.empty((0,2))
        running_outputs, running_outputs_prob, running_labels, running_targets = np.empty((0,2)), np.empty((0,2)), np.empty((0,1)), np.empty((0,1))
        running_folds = np.empty((0,1))
        running_informations = []
        for fold_test in fold_test_list:
            print('##################### Evaluating fold: ' + str(fold_test) + ' #####################')
            dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                            batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                            target_value=not classification, both_indicies=True, get_information=True,
                                                                                            train_samples=False, val_samples=False, test_samples=True)
            inputModel_path = ALLFOLD_MODELS_FOLDER + 'exp_fold_{}/'.format(fold_test) + 'model_best.pt'
            model_evaluation = torch.load(inputModel_path, map_location=device)
            n_samples, outputs, outputs_prob, labels, targets, informations = stat_mod_ut.get_data_single_fold(model_evaluation, dataloaders_dict['test'], debug=debug)
            folds = np.ones((n_samples,1)) * fold_test
            running_outputs, running_outputs_prob, running_labels, running_targets, running_folds, running_informations = np.concatenate((running_outputs, outputs)), np.concatenate((running_outputs_prob, outputs_prob)), np.concatenate((running_labels, labels)), np.concatenate((running_targets, targets)), np.concatenate((running_folds, folds)), running_informations+informations
                
        ## Aggregate quantities ##
        labels_predicted = np.expand_dims(np.argmax(running_outputs, axis=1),axis=1)
        aggregate_accuracy = accuracy_score(running_labels, labels_predicted)
        aggregate_confusionMatrix = confusion_matrix(running_labels, labels_predicted)
        aggregate_confusionMatrix_normalized = confusion_matrix(running_labels, labels_predicted, normalize='true')
        aggregate_spearmanr = spearmanr(running_targets, running_outputs[:,0])
        aggregate_spearmanr_prob = spearmanr(running_targets, running_outputs_prob[:,0])
        
        print('\nAggregate accuracy: {:.3f}, Aggregate spearmanr: {:.3f}, Aggregate spearmanr_prob: {:.3f}'.format(aggregate_accuracy, aggregate_spearmanr[0], aggregate_spearmanr_prob[0]))
        #############################

        ## Dataset creation ##
        data = np.concatenate((running_outputs, running_outputs_prob, labels_predicted, running_labels, running_targets, running_folds), axis=1)
        dataframe = DataFrame(data=data, columns=('nn_output_label0','nn_output_label1','nn_output_prob_label0','nn_output_prob_label1','label_prediction', 'label','EMOGAS index','fold'))
        information_columns = dataload_ut.get_columns_from_informationdict(running_informations)
        dataframe['bimbo_name'], dataframe['classe'], dataframe['esame_name'], dataframe['paziente'], dataframe['valore'], dataframe['video_name'], dataframe['processed_video_name'], dataframe['total_clip_frames'] = information_columns
        #############################

        ## Save results ##
        dataframe.to_csv(ALLFOLD_MODELS_FOLDER+'evaluation_dataframe.csv')
        f = open(ALLFOLD_MODELS_FOLDER + "evaluation_aggregate_metrics.txt", "w")
        f.write('Aggregate accuracy: {}\nAggregate spearmanr: {}, p_value: {}\nAggregate spearmanr_prob: {}, p_value: {}'.format(aggregate_accuracy, aggregate_spearmanr[0], aggregate_spearmanr[1], aggregate_spearmanr_prob[0], aggregate_spearmanr_prob[1]))
        f.write('\nConfusion matrix:\n' + str(aggregate_confusionMatrix))
        f.write('\nConfusion matrix normalized:\n' + str(aggregate_confusionMatrix_normalized))
        f.close()
        #############################
        
        


