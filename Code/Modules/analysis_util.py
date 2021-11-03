#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 01:25:01 2021

@author: luigidamico
"""

import dataloader_utils as dataload_ut
import pandas as pd
from dataloader_utils import NUM_ROWS
import matplotlib.pyplot as plt
import pims
from scipy.io import loadmat
import numpy as np
from config import ALLFOLD_MODELS_FOLDER, device, DATASET_PATH, DATASET_RAW_PATH
import pandas as pd
import torch
import dataloader_utils as dataload_ut
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import NoiseTunnel
import pickle
from dataloader_utils import train_img_transform
from dataloader_utils_old import train_transform as train_img_transform_old

#%%

def create_dataframe_train(out_path='../../Experiments/train_dataset.csv', target_value=False):
    '''
    Create the dataframe containing information about all the train batches for all folds
    load using: dataframe_train = pd.read_csv('../../Experiments/train_dataset.csv').drop('Unnamed: 0', axis=1)
    '''
    train_dict_info = {
        'bimbo_name': [],
        'classe': [],
        'paziente': [],
        'valore': [],
        'video_name': [],
        'processed_video_name': [],
        'fold': []
        }
    
    for fold_test in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        dataloaders_dict, _ = dataload_ut.get_mat_dataloaders_v2(['BEST', 'RDS'], basePath=DATASET_PATH, num_workers=0, fold_test=fold_test,
                                                                                        batch_size=32, mode='random_frame_from_clip', replicate_minority_classes=False, replicate_all_classes=0,
                                                                                        target_value=target_value, both_indicies=True, get_information=True,
                                                                                        train_samples=True, val_samples=False, test_samples=False)
        
        for batch in dataloaders_dict['train']:
            train_dict_info['bimbo_name'] = train_dict_info['bimbo_name'] + batch[3]['bimbo_name']
            train_dict_info['classe'] = train_dict_info['classe'] + batch[3]['classe']
            train_dict_info['paziente'] = train_dict_info['paziente'] + batch[3]['paziente']
            train_dict_info['valore'] = train_dict_info['valore'] + batch[3]['valore']
            train_dict_info['video_name'] = train_dict_info['video_name'] + batch[3]['video_name']
            train_dict_info['processed_video_name'] = train_dict_info['processed_video_name'] + batch[3]['processed_video_name']
            train_dict_info['fold'] = train_dict_info['fold'] + [fold_test] * len(batch[0])
        
    dataframe_train = pd.DataFrame(train_dict_info)
    dataframe_train.to_csv(out_path)
    return dataframe_train


def create_dataframe_video_byFold(dataframe_train, out_path=None):
    dataframe_videos = dataframe_train.groupby(['bimbo_name', 'classe', 'video_name', 'processed_video_name', 'fold'], as_index=False).sum()
    dataframe_videos['ospedale'] = create_ospedale_column(dataframe_videos)
    dataframe_videos['filename'] = create_filename_column(dataframe_videos)
    if out_path:
        dataframe_videos.to_csv(out_path)
    return dataframe_videos


def create_dataframe_videos(dataframe, out_path=None):
    '''
    Create the dataframe containing all the videos information. Each row is a video.
    dataframe_train: dataframe generated by create_and_save_dataframe_train function
    '''
    dataframe_videos = dataframe.groupby(['bimbo_name', 'classe', 'video_name', 'processed_video_name'], as_index=False).sum()
    dataframe_videos.drop('fold', axis=1, inplace=True)
    dataframe_videos['ospedale'] = create_ospedale_column(dataframe_videos)
    dataframe_videos['filename'] = create_filename_column(dataframe_videos)
    if out_path:
        dataframe_videos.to_csv(out_path)
    return dataframe_videos


def create_and_save_dataframe_patients(dataframe_videos, out_path='../../Experiments/patients_dataset.csv'):
    '''
    Create the dataframe containing all the patients information. Each row is a patient.
    dataframe_videos: dataframe generated by create_and_save_dataframe_videos function
    '''
    dataframe_videos['Num_video'] = 1
    dataframe_patients = dataframe_videos.groupby(['bimbo_name','ospedale','classe'], as_index=False).sum()
    dataframe_patients.drop(['valore'], axis=1, inplace=True)
    return dataframe_patients


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

def create_filename_column(dataset):
    filename_col = pd.Series(index=dataset.index, dtype=object)
    for idx in dataset.index:
        processed_video_name = dataset.loc[idx]['processed_video_name']
        filename_col.loc[idx] = processed_video_name.split('/')[1]
    return filename_col


def evaluate_dataset(dataset):
    '''
    Split the dataset into wrongPrediction and rightPrediction datasets
    '''
    dataset_wrongPrediction = dataset[dataset['label_prediction'] != dataset['label']]
    dataset_rightPrediction = dataset[dataset['label_prediction'] == dataset['label']]
    accuracy = len(dataset_rightPrediction) / len(dataset)
    return (dataset_rightPrediction, dataset_wrongPrediction), accuracy


def create_dataset_videoLevel(dataset, th=0.5):
    '''
    Create dataset at video Level
    '''
    dataset_videoLevel = dataset.groupby(['video_name','ospedale','bimbo_name','classe'], as_index=False).mean().set_index('video_name')
    dataset_videoLevel['label_prediction'] = (dataset_videoLevel['nn_output_prob_label1'] > th).astype('float')
    dataset_videoLevel_wrong_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction'] != dataset_videoLevel['label']]
    dataset_videoLevel_right_prediction = dataset_videoLevel[dataset_videoLevel['label_prediction'] == dataset_videoLevel['label']]
    accuracy = len(dataset_videoLevel_right_prediction) / len(dataset_videoLevel)
    print('Accuracy videoLevel: {:.3f}'.format(accuracy))
    return (dataset_videoLevel, dataset_videoLevel_right_prediction, dataset_videoLevel_wrong_prediction), accuracy


def analyze_one_video_prediction(dataset, idx):
    '''
    Plot frames of a specific video.
    Shows which one is correctly or wrongly predicted
    '''
    dataset_wrong_prediction = dataset[dataset['label_prediction'] != dataset['label']]
    dataset_right_prediction = dataset[dataset['label_prediction'] == dataset['label']]
    sample = dataset.loc[idx]
    video_name = sample['video_name']
    print("Taking video: "+video_name)
    wrong_prediction = dataset_wrong_prediction[dataset_wrong_prediction['video_name'] == video_name]
    right_prediction = dataset_right_prediction[dataset_right_prediction['video_name'] == video_name]
    n_frame_wrong = len(wrong_prediction)
    n_frame_right = len(right_prediction) 
    print('{} frames predicted wrong out of {} total'.format(n_frame_wrong, n_frame_wrong+n_frame_right))
    
    ### Load raw img   
    if sample['classe'] == 'BEST':
        clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' + sample['video_name']
    else:
        clip_raw_path = DATASET_RAW_PATH + '/' + sample['classe'] + '/' + sample['bimbo_name'] + '/' +  sample['esame_name'] + '/' + sample['video_name']
    try:
        clip_raw = pims.Video(clip_raw_path)
        img_raw = clip_raw[-1]
        plt.imshow(img_raw), plt.show()
    except:
        print('No image raw found')
    
    ### load mat imgs
    clip_mat_path = DATASET_PATH + '/' + sample['processed_video_name']
    matdata = loadmat(clip_mat_path)
    
    ### images showing
    for idx in wrong_prediction.index:
        k = wrong_prediction.loc[idx]['frame_key']
        prob_label0 = wrong_prediction.loc[idx]['nn_output_prob_label0']
        correct_label = wrong_prediction.loc[idx]['classe']
        fig = plt.imshow(matdata[k][:NUM_ROWS]), plt.title('{}, wrong prediction, correct: {}, prob_label0: {:.2f}'.format(k, correct_label, prob_label0)), plt.show()
    
    for idx in right_prediction.index:
        k = right_prediction.loc[idx]['frame_key']
        prob_label0 = right_prediction.loc[idx]['nn_output_prob_label0']
        correct_label = right_prediction.loc[idx]['classe']
        fig = plt.imshow(matdata[k][:NUM_ROWS]), plt.title('{}, correct prediction, correct: {}, prob_label0: {:.2f}'.format(k, correct_label, prob_label0)), plt.show()


def get_video_frame(dataset, idx, show=True, out_file_path=None):
    sample = dataset.loc[idx]
    clip_mat_path = DATASET_PATH + '/' + sample['processed_video_name']
    matdata = loadmat(clip_mat_path)
    k = sample['frame_key']
    img = matdata[k][:NUM_ROWS]
    if show:
        plt.imshow(img)
        plt.axis('off')
        if out_file_path is not None:
            plt.savefig(out_file_path, dpi=100)
    return img
    

def show_augmentations(dataset, idx, num_aug=2, out_file_path=None, old_transformation=False):
    img = get_video_frame(dataset, idx, show=False)
    plt.subplot(3,3,1)
    plt.imshow(img), plt.axis('off'), plt.title('Original image')
    transf = train_img_transform(NUM_ROWS) if not old_transformation else train_img_transform_old(NUM_ROWS)
    for i in range(2,num_aug+2):
        aug_img = transf(image=img)['image'] if not old_transformation else transf(img)
        aug_img_rescaled = ((aug_img - aug_img.min()) / (aug_img.max() - aug_img.min())).permute(1,2,0)
        idx_plot = i if i<=3 else i+1
        idx_plot = idx_plot if idx_plot <= 6 else idx_plot+1
        plt.subplot(3,3,idx_plot)
        plt.imshow(aug_img_rescaled), plt.axis('off'), plt.title('Augmented image, ex.'+str(i-1), fontsize='small')

    if out_file_path:
        plt.savefig(out_file_path, dpi=300)


def show_sample_attribution(dataset, idx, n_steps, show_original_img=True, out_file_path=None):
    n_frame = int(idx[-1])
    sample = dataset.loc[idx]
    fold_test = int(sample['fold'])
    clip_mat_path = DATASET_PATH + '/' + sample['processed_video_name']
    mat_data = dataload_ut.default_mat_loader(clip_mat_path, mode='entire_clip')
    img = mat_data[0][n_frame]
    input = dataload_ut.test_img_transform(dataload_ut.NUM_ROWS)(image=img)['image'].unsqueeze(dim=0)
    input_model_path = '../' + ALLFOLD_MODELS_FOLDER + 'exp_fold_{}/'.format(fold_test) + 'model_best.pt'
    model = torch.load(input_model_path, map_location=device)
    output = model(input)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label = torch.topk(output, 1)

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label, n_steps=n_steps)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')], N=256)
    attributions_ig_mod = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0))
    transformed_img = input*0.1435 + 0.1250
    transformed_img_mod = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
    if show_original_img:
        fig, _ = viz.visualize_image_attr_multiple(attributions_ig_mod,
                                              transformed_img_mod,
                                              ["original_image", "heat_map"],
                                              ["all", "positive"],
                                              cmap=default_cmap,
                                              show_colorbar=True)
    else:
        fig, _ = viz.visualize_image_attr(attributions_ig_mod,
                                              transformed_img_mod,
                                              method="heat_map",
                                              sign="positive",
                                              cmap=default_cmap,
                                              show_colorbar=True)
            
    if out_file_path is not None:
        fig.savefig(out_file_path, dpi=200, transparent=True)
    
    return attributions_ig


def show_training_histories_byFold(hists_path='/Users/luigidamico/Desktop/Thesis/Code/My code/repos/DL_QLUS/Experiments/experiment_allfold_exp_3/'):
    for fold in [0,1,2,3,4,5,6,7,8,9]:
        print('fold: ', fold)
        hist_path = hists_path + 'exp_fold_'+str(fold)+'/hist.pkl'
        hist_data = pickle.load(open(hist_path,'rb'))
        epoch_best_valLoss = np.array(hist_data[0][1][1]).argmin()
        print('hist best val loss: ', hist_data[0][1][1][epoch_best_valLoss])
        print('hist best test acc: ', hist_data[1][1][2][epoch_best_valLoss])
        print('hist best epoch: ', epoch_best_valLoss+1)
        print('hist last test acc: ', hist_data[1][1][2][-1])
        print()
        
        
def print_dataframe_stats(dataframe):
    for ospedale in dataframe['ospedale'].unique():
        df_ospedale = dataframe[dataframe['ospedale']==ospedale]
        for classe in dataframe['classe'].unique():
            df = df_ospedale[df_ospedale['classe']==classe]
            print('Ospedale: '+ospedale+'\tClasse: ',classe, '\ttot samples: ', len(df)) 


def print_dataset_counts(dataset):
    print(accuracy_score(dataset['label'], dataset['label_prediction']))
    print(confusion_matrix(dataset['label'], dataset['label_prediction']))
    print('------------------Naples') 
    dataset_naples = dataset[dataset['ospedale']=='Naples']
    print(accuracy_score(dataset_naples['label'], dataset_naples['label_prediction']))
    print(confusion_matrix(dataset_naples['label'], dataset_naples['label_prediction'], normalize='true'))
    print(confusion_matrix(dataset_naples['label'], dataset_naples['label_prediction']))
    print('------------------Florence')
    dataset_florence = dataset[dataset['ospedale']=='Florence']
    print(accuracy_score(dataset_florence['label'], dataset_florence['label_prediction']))
    print(confusion_matrix(dataset_florence['label'], dataset_florence['label_prediction'], normalize='true'))
    print(confusion_matrix(dataset_florence['label'], dataset_florence['label_prediction']))
    print('------------------Milan')
    dataset_milan = dataset[dataset['ospedale']=='Milan']
    print(accuracy_score(dataset_milan['label'], dataset_milan['label_prediction']))
    print(confusion_matrix(dataset_milan['label'], dataset_milan['label_prediction'], normalize='true'))
    print(confusion_matrix(dataset_milan['label'], dataset_milan['label_prediction']))
            
            
            