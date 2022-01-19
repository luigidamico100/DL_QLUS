#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:31:40 2021

@author: luigidamico
"""

import dataloader_utils as dataload_ut
import static_model_utils as stat_mod_ut
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
import pandas as pd
from scipy.stats import spearmanr
import os
np.set_printoptions(4)


def get_metrics_df(predictions_df):
    accuracy = accuracy_score(predictions_df['label'], predictions_df['label_prediction'])
    conf_matrix = confusion_matrix(predictions_df['label'], predictions_df['label_prediction'])
    conf_matrix_norm = confusion_matrix(predictions_df['label'], predictions_df['label_prediction'], normalize='true')
    spearman_corr = spearmanr(predictions_df['nn_output_prob_label0'], predictions_df['valore'])[0]
    metrics_list = [accuracy, conf_matrix, conf_matrix_norm, spearman_corr]
    return np.array(metrics_list, dtype=object)


def get_results_df_hospital(predictions_df):
    predictions_df_naples = predictions_df[predictions_df['ospedale']=='Naples']
    predictions_df_florence = predictions_df[predictions_df['ospedale']=='Florence']
    predictions_df_milan = predictions_df[predictions_df['ospedale']=='Milan']
    results_df_hospital = pd.DataFrame(columns=['Accuracy', 'Conf_matrix', 'Conf_matrix_norm', 'Spearman_corr'])
    results_df_hospital.loc['Naples'] = get_metrics_df(predictions_df_naples)
    results_df_hospital.loc['Florence'] = get_metrics_df(predictions_df_florence)
    results_df_hospital.loc['Milan'] = get_metrics_df(predictions_df_milan)
    results_df_hospital.loc['Overall'] = get_metrics_df(predictions_df)
    return results_df_hospital


def create_ospedale_column(dataset):
    ospedale_col = pd.Series(index=dataset.index, dtype=object)
    for idx in dataset.index:
        bimbo_name = dataset.loc[idx]['bimbo_name']
        if 'Buzzi' in bimbo_name:
            ospedale_col.loc[idx] = 'Milan'
        elif 'Firenze' in bimbo_name:
            ospedale_col.loc[idx] = 'Florence'
        else:
            ospedale_col.loc[idx] = 'Naples'
    return ospedale_col

def create_dataset_videoLevel(dataset, th=0.5):
    '''
    Create dataset at video Level
    '''
    dataset_videoLevel = dataset.groupby(['video_name','ospedale','bimbo_name','classe','valore'], as_index=False).mean().set_index('video_name')
    dataset_videoLevel['label_prediction'] = (dataset_videoLevel['nn_output_prob_label1'] > th).astype('float')
    dataset_videoLevel_wrong_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction'] != dataset_videoLevel['label']]
    dataset_videoLevel_right_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction'] == dataset_videoLevel['label']]
    accuracy = len(dataset_videoLevel_right_prediction) / len(dataset_videoLevel)
    print('Accuracy videoLevel: {:.3f}'.format(accuracy))
    return (dataset_videoLevel, dataset_videoLevel_right_prediction, dataset_videoLevel_wrong_prediction), accuracy


#%% Evaluate model
if __name__ == '__main__':
    loss, metric = config.get_problem_stuff()
    os.mkdir(ALLFOLD_MODELS_FOLDER+'results/')
    
    print('##################### Evaluating all models in: ' + ALLFOLD_MODELS_FOLDER + ' #####################\n')
    for phase in ['val', 'test']:
        print('##################### Evaluating phase: ' + phase + ' #####################')
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
                                                                                            train_samples=False, val_samples=(phase=='val'), test_samples=(phase=='test'))
            inputModel_path = ALLFOLD_MODELS_FOLDER + 'exp_fold_{}/'.format(fold_test) + 'model_best.pt'
            model_evaluation = torch.load(inputModel_path, map_location=device)
            n_samples, outputs, outputs_prob, labels, targets, informations = stat_mod_ut.get_data_single_fold(model_evaluation, dataloaders_dict[phase], debug=debug)
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
        predictions_df = DataFrame(data=data, columns=('nn_output_label0','nn_output_label1','nn_output_prob_label0','nn_output_prob_label1','label_prediction', 'label','EMOGAS index','fold'))
        information_columns = dataload_ut.get_columns_from_informationdict(running_informations)
        predictions_df['bimbo_name'], predictions_df['classe'], predictions_df['esame_name'], predictions_df['paziente'], predictions_df['valore'], predictions_df['video_name'], predictions_df['processed_video_name'], predictions_df['frame_key'], predictions_df['total_clip_frames'], predictions_df['keys'] = information_columns
        predictions_df.set_index('keys', inplace=True)
        predictions_df['ospedale'] = create_ospedale_column(predictions_df)
        predictions_df.info()
        (predictions_video_df, _, _), _ = create_dataset_videoLevel(predictions_df, th=0.5)
        results_df_hospital = get_results_df_hospital(predictions_df)
        results_video_df_hospital = get_results_df_hospital(predictions_video_df)
        #############################

        ## Save results ##
        predictions_df.to_csv(ALLFOLD_MODELS_FOLDER+'results/'+phase+'_predictions_df.csv', float_format='%.4f')
        results_df_hospital.to_csv(ALLFOLD_MODELS_FOLDER+'results/'+phase+'_results_df_hospital.csv', float_format='%.4f')
        results_video_df_hospital.to_csv(ALLFOLD_MODELS_FOLDER+'results/'+phase+'_results_video_df_hospital.csv', float_format='%.4f')
        #############################


