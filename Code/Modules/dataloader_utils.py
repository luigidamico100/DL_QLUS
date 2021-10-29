#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:00:30 2021

@author: luigidamico
"""

from random import gauss, randint, choice, sample, seed
from scipy.io import loadmat
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import DatasetFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import ToFloat
import cv2
import math


#%%

NUM_ROWS = 224
NUM_COLUMNS = 461
# NUM_COLUMNS = 200
NUM_FRAMES = 6

all_classes = ['BEST', 'RDS', 'TTN']

CV_FOLD = {'BEST': [[1, 2, 68, 75, 85],
                    [3, 4, 21, 76, 86],
                    [5, 6, 69, 77, 87],
                    [7, 8, 70, 78],
                    [9, 10, 71, 79],
                    [11, 12, 22, 80],
                    [13, 14, 72, 81],
                    [15, 16, 73, 82],
                    [17, 18, 74, 83],
                    [19, 20, 23, 84]],
           'RDS': [[24, 33, 51],
                   [25, 38, 53],
                   [26, 35, 49],
                   [34, 44, 52],
                   [27, 40, 45],
                   [30, 43, 47],
                   [28, 36, 39],
                   [29, 37, 50],
                   [31, 41, 46],
                   [32, 42, 48]],
           'TTN': [[54, 64],
                   [55],
                   [56],
                   [57],
                   [58, 65],
                   [59],
                   [60],
                   [61],
                   [62, 66],
                   [63]]}

train_img_transform = lambda num_rows : A.Compose([
    #input: numpy[(H, W, C)]
    A.Rotate(limit=10, p=1.0, border_mode=(cv2.BORDER_CONSTANT)),
    A.RandomResizedCrop(height=num_rows, width=NUM_COLUMNS, scale=(0.99, 1.0), ratio=(0.99, 1.01)),
    # A.Resize(height=num_rows, width=NUM_COLUMNS),
    # A.RandomCrop(height=num_rows, width=NUM_COLUMNS),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(.25, .25, p=.5),          #it raise an error for video-mode
    # A.RandomBrightnessContrast(p=0.5),
    ToFloat(max_value=(255)),
    A.Normalize(mean = 0.1250, std = 0.1435, max_pixel_value=1.0),
    # A.Normalize(mean = 0.5, std = 0.5, max_pixel_value=1.0),
    ToTensorV2(),
    # output: torch.Size([C, H, W])
])

test_img_transform = lambda num_rows : A.Compose([
    #input: numpy[(H, W, C)]
    A.Resize(height=num_rows, width=NUM_COLUMNS),
    # A.RandomCrop(height=num_rows, width=NUM_COLUMNS),
    ToFloat(max_value=(255)),
    A.Normalize(mean = 0.1250, std = 0.1435, max_pixel_value=1.0),
    ToTensorV2(),
    #output: torch.Size([C, H, W])
])


#%% Functions

def default_mat_loader(path, num_rows=NUM_ROWS, return_value=False, mode='fixed_number_of_frames', get_information=False):

    matdata = loadmat(path)
    valore = matdata['valore']
        
    frame_keys = []
    if mode == 'fixed_number_of_frames':
        count_frames = 0
        data = []
        for k in matdata.keys():
            if k.startswith('f') and len(k) < 3 and count_frames < NUM_FRAMES:
                count_frames += 1
                data.append(matdata[k][:num_rows])
                frame_keys.append(k)
        data = np.array(data)
    elif mode == 'fixed_number_of_frames_1ch':
        data = [matdata[k][:num_rows] for k in matdata.keys() if k.startswith('f') and len(k) < 3]
        data = data[:NUM_FRAMES]
        data = np.array(data)
        data = np.delete(data, 0, 3)
        data = np.delete(data, 0, 3)
    elif mode == 'entire_clip':
        data = []
        for k in matdata.keys():
            if k.startswith('f') and len(k) < 3:
                data.append(matdata[k][:num_rows])
                frame_keys.append(k)
        data = np.array(data)
    elif mode == 'random_frame_from_clip' or mode == 'random_frame_from_clip_old':
        f = choice([k for k in matdata.keys() if k.startswith('f') and len(k) < 3])
        data = matdata[f][:num_rows]

    
    if get_information:
        if len(frame_keys) > 0:
            processed_video_path = path.split('/')
            information_dit = [{
                'bimbo_name': str(matdata['bimbo_name'][0]),
                'classe': str(matdata['classe'][0]),
                'esame_name': str(matdata['esame_name'][0]),
                'paziente': str(matdata['paziente'][0][0]),
                'valore': str(matdata['valore'][0][0]),
                'video_name': str(matdata['video_name'][0]),
                'processed_video_name': processed_video_path[-2] + '/' + processed_video_path[-1],
                'frame_key': k, 
                'total_clip_frames': len(data)
                } for k in frame_keys]
        else:
            processed_video_path = path.split('/')
            information_dit = {
                'bimbo_name': str(matdata['bimbo_name'][0]),
                'classe': str(matdata['classe'][0]),
                'esame_name': str(matdata['esame_name'][0]),
                'paziente': str(matdata['paziente'][0][0]),
                'valore': str(matdata['valore'][0][0]),
                'video_name': str(matdata['video_name'][0]),
                'processed_video_name': processed_video_path[-2] + '/' + processed_video_path[-1],
                'frame_key': 'None', 
                'total_clip_frames': len(data)
            }
        return data, float(valore.item()/480.), information_dit
    else:
        return data, float(valore.item()/480.)


def balance_datasets(datasets):
    l = max([len(dataset) for dataset in datasets])
    for dataset in datasets:
        while len(dataset) < l:
            dataset.samples += sample(dataset.samples, min(len(dataset), l - len(dataset)))
    return datasets


def replicate_datasets(datasets, increase_factor=0):
    if increase_factor > 1:
        for dataset in datasets:
            dataset.samples = dataset.samples * increase_factor
    return datasets


class BalanceConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        datasets = balance_datasets(datasets)
        super(BalanceConcatDataset, self).__init__(datasets)


class LUSFolder(DatasetFolder):  # eredita da DatasetFolder
    def __init__(self, root, train_phase, target_value=False, both_indicies=False, mode='fixed_number_of_frames', subset_in=None, subset_out=None, num_rows=224,
                 subset_var='paziente', exclude_class=None, exclude_val_higher=None, random_seed=0, loader=None, get_information=False):
        seed(random_seed)
        self.target_value = target_value
        self.both_indicies = both_indicies
        self.get_information = get_information
        self.mode = mode
        self.transform = train_img_transform(num_rows) if train_phase else test_img_transform(num_rows)

        if loader is None:
            if train_phase:
                loader = lambda path: default_mat_loader(path, num_rows, return_value=target_value, mode=mode, get_information=get_information)
            else:
                loader = lambda path: default_mat_loader(path, num_rows, return_value=target_value, mode=mode, get_information=get_information)
        # definisce la funzione che seleziona i file in base a subset_in e subset_out
        assert subset_in is None or subset_out is None  # almeno uno dei due deve essere None
        if subset_in is not None:
            # prende i file dei soggetti presenti in subset_in
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] in subset_in
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] in subset_in \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher
        elif subset_out is not None:
            # prende i file dei soggetti non presenti in subset_in
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] not in subset_out
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names=subset_var)[subset_var] not in subset_out \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher
        else:
            # prende tutti i file
            if exclude_val_higher is None:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class]))
            else:
                is_valid_fun = lambda path: path.endswith('.mat') and (exclude_class is None or all([e not in path for e in exclude_class])) \
                                            and loadmat(path, variable_names='valore')['valore'] < exclude_val_higher

        super(LUSFolder, self).__init__(root=root, loader=loader, extensions=None,
                                        transform=self.transform, target_transform=None, is_valid_file=is_valid_fun)
        self.imgs = self.samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, label = self.samples[index]
        if self.get_information:
            sample, target, informations = self.loader(path)
        else:
            sample, target = self.loader(path)
        # target = torch.tensor(target).type(torch.LongTensor)
        
        if self.transform is not None:
            # print('\tsample.shape = ', sample.shape)
            if self.mode=='entire_clip' or self.mode == 'fixed_number_of_frames' or self.mode == 'fixed_number_of_frames_1ch':
                sample = sample.transpose(1,2,0,3)
                H = sample.shape[0]
                W = sample.shape[1]
                T = sample.shape[2]
                C = sample.shape[3]
                sample = np.reshape(sample, (H, W, T*C))
                sample = self.transform(image=sample)['image']
                H = sample.shape[1] #H and W could change after transformation
                W = sample.shape[2]
                sample = torch.reshape(sample, (T, C, H, W))
                # print('\tsample transformed.shape = ', sample.shape)
            elif self.mode == 'random_frame_from_clip' or self.mode == 'random_frame_from_clip_old':
                sample = self.transform(image=sample)['image']
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print('\tout sample.shape = ', sample.shape)
        if self.both_indicies:
            if self.get_information:
                return sample, label, target, informations
            else:
                return sample, label, target
        else:
            if self.target_value:
                return sample, target
            else:
                return sample, label


def show_images(data_loader, batches=10):
    # salvataggio immagini per controllo
    from os import makedirs
    from shutil import rmtree
    save_img = lambda tensor, name: ToPILImage()((tensor.numpy().transpose((1, 2, 0)).squeeze() * 255.)  # w/o Normalization
                                                  .astype('uint8')).save(name)
    # save_img = lambda tensor, name: ToPILImage()((tensor.numpy().transpose((1, 2, 0)).squeeze() * 128. + 128.)  # w/ Normalization
    #                                              .clip(0, 255).astype('uint8')).save(name)
    try: rmtree('./temp')
    except: pass
    makedirs('./temp', exist_ok=True)
    for b, (X, y) in enumerate(data_loader):
        for i, x in enumerate(X):
            save_img(x, 'temp/batch%d_img%d_c%d.png' % (b, i, y[i].item()))
        print('batch', b, 'written')
        if b >= batches:
            break


def get_mat_dataloaders_v2(classes, basePath, target_value=False, both_indicies=False, replicate_minority_classes=True, fold_test=0, fold_val=None, batch_size=32, num_workers=4, replicate_all_classes=10, mode='fixed_number_of_frames',
                           train_samples=True, val_samples=True, test_samples=True, get_information=False):
    
    # get_information = get_information if (both_indicies and not train_samples) else False   #This cases have not been implemented
    get_information = get_information if (both_indicies) else False   #This cases have not been implemented
    
    print('\n\n---------- Creating datasets and dataloaders ----------')
    if fold_val is None: fold_val = fold_test - 1
    print('Validation fold:', fold_val, ', Test fold:', fold_test)
    if len(classes) == 2 and all_classes[-1] in classes:
        Warning('Correggere le label del dataset!')

    # creazione dataset per classe con selezione degli utenti
    train_ds, val_ds, test_ds = [], [], []
    for class_name in classes:
        print('\t- creating data sets for class', class_name)
        exclude_class = all_classes.copy()
        exclude_class.remove(class_name)
        
        if train_samples:
            train_ds.append(LUSFolder(root=basePath, train_phase=True, target_value=target_value, both_indicies=both_indicies, mode=mode, subset_out=CV_FOLD[class_name][fold_test] + CV_FOLD[class_name][fold_val],
                                      exclude_class=exclude_class, num_rows=NUM_ROWS, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None, get_information=get_information))
            print('\t\t- TRAIN n. samples founded: ', len(train_ds[-1]))
            
        mode_test = 'fixed_number_of_frames' if mode=='random_frame_from_clip' else mode
        
        if val_samples:
            val_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, both_indicies=both_indicies, mode=mode_test, subset_in=CV_FOLD[class_name][fold_val],
                                    num_rows=NUM_ROWS, exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None, get_information=get_information))
            print('\t\t- VAL n. samples founded: ', len(val_ds[-1]))
            
        if test_samples:            
            test_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, both_indicies=both_indicies, mode=mode_test, subset_in=CV_FOLD[class_name][fold_test],
                                     num_rows=NUM_ROWS, exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None, get_information=get_information))
            print('\t\t- TEST n. samples founded: ', len(test_ds[-1]))            


    # bilanciamento del numero di campioni delle classi ed eventuale replicazione di tutti i campioni
    if train_samples and replicate_minority_classes:
        print('\t- balancing data sets by sample duplications (replicate_all_classes=%d)' % replicate_all_classes)
        print('\t  - before:', [len(_) for _ in train_ds])
        train_ds = balance_datasets(train_ds)
        print('\t  - after:', [len(_) for _ in train_ds])
    if train_samples and replicate_all_classes > 1:
        print('\t- replicating train data sets by sample duplications %d times' % replicate_all_classes)
        print('\t  - before:', [len(_) for _ in train_ds])
        train_ds = replicate_datasets(train_ds, replicate_all_classes)
        print('\t  - after:', [len(_) for _ in train_ds])

    def collate_fn(data):  # collate all frames of a validation or test video
                            # data: list of batch_size//... tuple (batch). Each tuple contains a list of the img frames (as tensors) and the label        
        if both_indicies:
            if get_information:
                X, Y1, Y2, INF = [], [], [], []
                for x, y1, y2, inf in data:
                    X += x
                    Y1 += [y1] * len(x)
                    Y2 += [y2] * len(x)
                    INF += inf
                Y1 = torch.Tensor(Y1)
                Y2 = torch.Tensor(Y2)
                X = torch.stack(X)
                Y1 = Y1.long()
                return X, Y1, Y2, INF
            else:
                X, Y1, Y2 = [], [], []
                for x, y1, y2 in data:
                    X += x
                    Y1 += [y1] * len(x)
                    Y2 += [y2] * len(x)
                Y1 = torch.Tensor(Y1)
                Y2 = torch.Tensor(Y2)
                X = torch.stack(X)
                Y1 = Y1.long()
                return X, Y1, Y2
        else:
            X, Y = [], []
            for x, y in data:
                X += x
                Y += [y] * len(x)
            Y = torch.Tensor(Y)
            X = torch.stack(X)
            if not target_value:
                Y = Y.long()
            return X, Y

    # data loader sulla concatenazione dei dataset delle singole classi
    train_dl, val_dl, test_dl = None, None, None
    if train_samples:
        train_dl = DataLoader(ConcatDataset(train_ds), num_workers=num_workers, pin_memory=True,
                              shuffle=True, batch_size=batch_size)
        print('\t- Train num samples: {}, Required batch iteration: {:.1f}'.format(sum([len(_) for _ in train_ds]), sum([len(_) for _ in train_ds]) / batch_size))
        
    if mode == 'random_frame_from_clip':
        collate = collate_fn
        batch_size = math.ceil(batch_size/6)
    else:
        collate = None
        
    if val_samples:
        val_dl = DataLoader(ConcatDataset(val_ds), num_workers=num_workers, pin_memory=True,
                            shuffle=True, batch_size=batch_size, collate_fn=collate)
        
    if test_samples:
        test_dl = DataLoader(ConcatDataset(test_ds), num_workers=num_workers, pin_memory=True,
                             shuffle=True, batch_size=batch_size, collate_fn=collate) 

    print()
    dataloaders_dict = {'train' : train_dl, 'val' : val_dl, 'test' : test_dl}
    datasets_dict = {'train' : train_ds, 'val' : val_ds, 'test' : test_ds}
    
    return dataloaders_dict, datasets_dict


def get_columns_from_informationdict(all_informations):
    col_bimbo_name = [None] * len(all_informations)
    col_classe = [None] * len(all_informations)
    col_esame_name = [None] * len(all_informations)
    col_paziente = [None] * len(all_informations)
    col_valore = [None] * len(all_informations)
    col_video_name = [None] * len(all_informations)
    col_processed_video_name = [None] * len(all_informations)
    col_frame_key = [None] * len(all_informations)
    col_total_clip_frames = [None] * len(all_informations)
    col_keys = [None] * len(all_informations)

    for idx, informations in enumerate(all_informations):    
        col_bimbo_name[idx] = informations['bimbo_name']
        col_classe[idx] = informations['classe']
        col_esame_name[idx] = informations['esame_name']
        col_paziente[idx] = informations['paziente']
        col_valore[idx] = informations['valore']
        col_video_name[idx] = informations['video_name']
        col_processed_video_name[idx] = informations['processed_video_name']
        col_frame_key[idx] = informations['frame_key']
        col_total_clip_frames[idx] = informations['total_clip_frames']
        col_keys[idx] = informations['processed_video_name'] + '\t' + informations['frame_key']
    
    return col_bimbo_name, col_classe, col_esame_name, col_paziente, col_valore, col_video_name, col_processed_video_name, col_frame_key, col_total_clip_frames, col_keys



#%%
classification_classes = ['BEST', 'RDS']
DATASET_PATH  = '/Volumes/SD Card/ICPR/Dataset_processato/Dataset_f'
num_workers = 0
fold_test = 0
batch_size = 16
# Mode to choose from [random_frame_from_clip_old, random_frame_from_clip, fixed_number_of_frames, fixed_number_of_frames_1ch]
mode = 'random_frame_from_clip'
replicate_all_classes = 1
classification = True
both_indicies = True
get_information = True

if __name__ == '__main__':
    
    dataloaders_dict, datasets_dict = get_mat_dataloaders_v2(classification_classes, basePath=DATASET_PATH, num_workers=num_workers, fold_test=fold_test,
                                                                                   batch_size=batch_size, mode=mode, replicate_all_classes=replicate_all_classes,
                                                                                   target_value=not classification, both_indicies=both_indicies, get_information=get_information,
                                                                                   train_samples=True, val_samples=False, test_samples=True)


    # train_dl = dataloaders_dict['train']
    # train_it = iter(train_dl)
    # train_get = next(train_it)
    # val_ds = datasets_dict['val']
    # a = val_ds[0] 
    # a.mode
          
    
    test_dl = dataloaders_dict['test']
    test_it = iter(test_dl)
    test_get = next(test_it)
    print(test_get[0].shape)
    print(test_get[1])
    if both_indicies:
        print(test_get[2])         
        
    train_dl = dataloaders_dict['train']
    train_it = iter(train_dl)
    train_get = next(train_it)
    print(train_get[0].shape)
    print(train_get[1])
    if both_indicies:
        print(train_get[2])
    
    
    val_dl = dataloaders_dict['val']
    val_it = iter(val_dl)
    val_get = next(val_it)
    print(val_get[0].shape)
    print(val_get[1])
    if both_indicies:
        print(val_get[2])

    
    # train_dl, val_dl, test_dl = dataloaders_dict['train'], dataloaders_dict['val'], dataloaders_dict['test']
    # train_it, val_it, test_it = iter(train_dl), iter(val_dl), iter(test_dl)
    # train_get, val_get, test_get = next(train_it), next(val_it), next(test_it)
    
    

