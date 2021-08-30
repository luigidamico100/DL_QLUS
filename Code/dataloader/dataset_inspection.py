#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:11:25 2021

@author: luigidamico
"""
from dataloader.dataloader_utils import default_mat_loader, NUM_ROWS, NUM_COLUMNS
import matplotlib.pyplot as plt
import numpy as np


labels_map = {'BEST':0, 'RDS':1, 'TTN':2}
labels_map = {0:'BEST', 1:'RDS', 2:'TTN'}

#%%

def plot_imgs(imgs, orig_img=None, row_title=None, **imshow_kwargs):
    '''
    some description...

    Parameters
    ----------
    imgs : list of list. images have to be as shape (C,H,W) (0-1 float or 0-255 int)
        DESCRIPTION.

    '''
    #https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + (orig_img is not None)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if (orig_img is not None) else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if (orig_img is not None):
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def prot_frames_from_dataloader(dataloader, num_batches=10):
    '''
    for dataloader obtained in mode 'random_frame_from_clip'
    Using dataloader means there output is the transformed version of the original input
    '''
    
    dataloader_it = iter(dataloader)
    batch_x, batch_y = next(dataloader_it)
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        if batch_idx > num_batches:
            break
    
        imgs = list(batch_x)
        labels = [labels_map[k.item()] for k in list(batch_y)]
    
        #https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]
            labels = [labels]
    
        num_rows = len(imgs)
        num_cols = len(imgs[0])
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                img = img.permute(1,2,0)  # (C, H, W) -> (H, W, C) (imshow condition)
                img = img * 0.1435 + 0.1250     # unNormalization
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img))
                ax.title.set_text(labels[row_idx][col_idx]) #check if is the opposite [col_idx][row_idx]
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
        plt.tight_layout()


def get_dataset_stats(dataloader):
    '''
    for dataloader obtained in mode 'random_frame_from_clip'
    Using dataloader means there output is the transformed version of the original input
    '''
    import torch
    channel = 0
    
    train_dl_it = iter(dataloader)
    batch = next(train_dl_it)[0]
    data_w_channels = batch
    data_wo_channels = batch[:,channel]
    for batch in train_dl_it:
        data_wo_channels = torch.cat((data_wo_channels, batch[0][:,channel]), dim=0)
        data_w_channels = torch.cat((data_w_channels, batch[0]), dim=0)
    mean_w_channels = data_w_channels.mean()    #tensor(0.1250)
    mean_wo_channels= data_wo_channels.mean()   #tensor(0.1244)
    std_w_channels = data_w_channels.std()      #tensor(0.1435)
    std_wo_channels= data_wo_channels.std()     #tensor(0.1439)
    
    return mean_w_channels, std_w_channels


def inspect_dataset(dataset):
    '''
    This should work for dataset generated in all modes
    The original .mat images are loaded (thus, no transformation is performed)
    '''
    mean = 0
    for path, label in dataset.samples:
        img = default_mat_loader(path, mode='random_frame_from_clip')
        # print('channel mean: ', img.mean(axis=(0,1)))
        # print('shape: ', img.shape.__str__())
        # plt.imshow(img);    plt.show()
        # if img.shape[1] < 400:
            # input("Press Enter to continue...")
        mean += img.mean()
    return mean
        

def get_stats_datasets_labels(datasets):
    means = [0]*3
    for idx, dataset in enumerate(datasets):
        for path, label in dataset.samples:
            img = default_mat_loader(path, mode='random_frame_from_clip')
            means[idx] += img.mean()
        means[idx] /= len(dataset.samples)
        print("mean for label %d: %.4f" %(idx, means[idx]))
    return means


#%% dataset visualization
prot_frames_from_dataloader(train_dl, num_batches=4)
prot_frames_from_dataloader(val_dl, num_batches=4)


            
#%% dataset statistics

# mean, std = get_dataset_stats(train_dl)
inspect_dataset(train_ds[0])
inspect_dataset(train_ds[1])

means = get_stats_datasets_labels(train_ds)





