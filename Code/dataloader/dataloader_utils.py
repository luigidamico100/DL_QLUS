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

#%%

NUM_ROWS = 224
NUM_COLUMNS = 461
NUM_FRAMES = 6

all_classes = ['BEST', 'RDS', 'TTN']

CV_FOLD = {'BEST': [[1, 2],
                    [3, 4, 21],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12, 22],
                    [13, 14],
                    [15, 16],
                    [17, 18],
                    [19, 20, 23]],
           'RDS': [[24, 33, 51],
                   [25, 38],
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
    A.RandomResizedCrop(height=num_rows, width=NUM_COLUMNS, scale=(0.99, 1.0), ratio=(0.99, 1.01)),
    # A.Resize(height=num_rows, width=NUM_COLUMNS),
    # A.RandomCrop(height=num_rows, width=NUM_COLUMNS),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=1.0, border_mode=(cv2.BORDER_CONSTANT)),
    A.ColorJitter(.15, .15),
    A.RandomBrightnessContrast(p=0.2),
    ToFloat(max_value=(255)),
    A.Normalize(mean = 0.1250, std = 0.1435, max_pixel_value=1.0),
    #A.Normalize(mean = 0.5, std = 0.5, max_pixel_value=1.0),
    ToTensorV2(),
    #output: torch.Size([C, H, W])
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

def default_mat_loader(path, num_rows=NUM_ROWS, return_value=False, mode='entire_clip'):

    data = loadmat(path)
    if return_value:
        valore = data['valore']
        
    if mode=='entire_clip':
        data = [data[k][:num_rows] for k in data.keys() if k.startswith('f') and len(k) < 3]
        data = data[:NUM_FRAMES]
        data = np.array(data)
    elif mode=='entire_clip_1ch':
        data = [data[k][:num_rows] for k in data.keys() if k.startswith('f') and len(k) < 3]
        data = data[:NUM_FRAMES]
        data = np.array(data)
        data = np.delete(data, 0, 3)
        data = np.delete(data, 0, 3)
    elif mode=='random_frame_from_clip':
        f = choice([k for k in data.keys() if k.startswith('f') and len(k) < 3])
        data = data[f][:num_rows]
        
    # print('\tloded mat shape: ', data.shape)

    if return_value:
        return data, float(valore.item()/480.)
    else:
        return data



def balance_datasets(datasets):
    # bilancia il numero di campioni nei vari dataset per replicazione
    l = max([len(dataset) for dataset in datasets])
    for dataset in datasets:
        while len(dataset) < l:
            dataset.samples += sample(dataset.samples, min(len(dataset), l - len(dataset)))
    return datasets


def replicate_datasets(datasets, increase_factor=0):
    # aumenta il numero di campioni di tutti i datset per replicazione per far durare di più l'epoca
    if increase_factor > 1:
        for dataset in datasets:
            dataset.samples = dataset.samples * increase_factor
    return datasets


class BalanceConcatDataset(ConcatDataset):  # eredita da ConcatDataset
    def __init__(self, datasets):
        datasets = balance_datasets(datasets)
        super(BalanceConcatDataset, self).__init__(datasets)


class LUSFolder(DatasetFolder):  # eredita da DatasetFolder
    def __init__(self, root, train_phase, target_value=False, mode='entire_clip', subset_in=None, subset_out=None, num_rows=224,
                 subset_var='paziente', exclude_class=None, exclude_val_higher=None, random_seed=0, loader=None):
        seed(random_seed)
        self.target_value = target_value
        self.mode = mode

        # definisce le trasformazioni da effettuare
        # if mode == 'random_frame_from_clip':
        self.transform = train_img_transform(num_rows) if train_phase else test_img_transform(num_rows)
        # elif mode == 'entire_clip':
            # self.transform = train_clip_transform(num_rows) if train_phase else test_clip_transform(num_rows)
        if loader is None:
            if train_phase:
                loader = lambda path: default_mat_loader(path, num_rows, return_value=target_value, mode=mode)
            else:
                loader = lambda path: default_mat_loader(path, num_rows, return_value=target_value, mode=mode)  #This should be always mode = 'entire_clip'
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

        # costruttore
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
        path, target = self.samples[index]
        # print('Getting item: ', path)
        if self.target_value:  # loader restituisce anche il valore target (sovrascritto alla label della classe)
            sample, target = self.loader(path)
        else:
            sample = self.loader(path)
        # target = torch.tensor(target).type(torch.LongTensor)
        if self.transform is not None:
            # print('\tsample.shape = ', sample.shape)
            if self.mode == 'entire_clip' or self.mode == 'entire_clip_1ch':
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
            elif self.mode == 'random_frame_from_clip':
                sample = self.transform(image=sample)['image']
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print('\tout sample.shape = ', sample.shape)
        return sample, target


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


def get_mat_dataloaders(classes, basePath, target_value=False, replicate_minority_classes=True, fold_test=0, fold_val=None, batch_size=32, num_workers=4, replicate_all_classes=10, mode='entire_clip'):
    if fold_val is None: fold_val = fold_test - 1
    print('Validation fold:', fold_val, '\nTest fold:', fold_test)
    if len(classes) == 2 and all_classes[-1] in classes:
        Warning('Correggere le label del dataset!')

    # creazione dataset per classe con selezione degli utenti
    train_ds, val_ds, test_ds = [], [], []
    for class_name in classes:
        print('- creating data sets for class', class_name)
        exclude_class = all_classes.copy()
        exclude_class.remove(class_name)
        train_ds.append(LUSFolder(root=basePath, train_phase=True, target_value=target_value, mode = mode, subset_out=CV_FOLD[class_name][fold_test] + CV_FOLD[class_name][fold_val],
                                  exclude_class=exclude_class, num_rows=NUM_ROWS, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None))
        val_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, mode = mode, subset_in=CV_FOLD[class_name][fold_val],
                                num_rows=NUM_ROWS, exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None))
        test_ds.append(LUSFolder(root=basePath, train_phase=False, target_value=target_value, mode = mode, subset_in=CV_FOLD[class_name][fold_test],
                                 num_rows=NUM_ROWS, exclude_class=exclude_class, exclude_val_higher=450 if not target_value and class_name != 'BEST' else None))
        print('  - found train/val/test samples:', len(train_ds[-1]), len(val_ds[-1]), len(test_ds[-1]))

    # bilanciamento del numero di campioni delle classi ed eventuale replicazione di tutti i campioni
    if replicate_minority_classes:
        print('- balancing data sets by sample duplications (replicate_all_classes=%d)' % replicate_all_classes)
        print('  - before:', [len(_) for _ in train_ds])
        train_ds = balance_datasets(train_ds)
        print('  - after:', [len(_) for _ in train_ds])
    if replicate_all_classes > 1:
        print('- replicating train data sets by sample duplications %d times' % replicate_all_classes)
        print('  - before:', [len(_) for _ in train_ds])
        train_ds = replicate_datasets(train_ds, replicate_all_classes)
        print('  - after:', [len(_) for _ in train_ds])

    # data loader sulla concatenazione dei dataset delle singole classi
    print('- creating data loaders')
    train_dl = DataLoader(ConcatDataset(train_ds), num_workers=num_workers, pin_memory=True,
                          shuffle=True, batch_size=batch_size)
    print('\t- Train num iteration to complete dataset: {:.1f}'.format(sum([len(_) for _ in train_ds]) / batch_size))
    val_dl = DataLoader(ConcatDataset(val_ds), num_workers=num_workers, pin_memory=True,
                        shuffle=True, batch_size=batch_size)    #batch_size=batch_size//5 originally
    print('\t- Val num iteration to complete dataset: {:.1f}'.format(sum([len(_) for _ in val_ds]) / batch_size))
    test_dl = DataLoader(ConcatDataset(test_ds), num_workers=num_workers, pin_memory=True,
                         shuffle=True, batch_size=batch_size)   #batch_size=batch_size//5 originally
    print('\t- Test num iteration to complete dataset: {:.1f}'.format(sum([len(_) for _ in test_ds]) / batch_size))
    
    dataloaders_dict = {'train' : train_dl, 'val' : val_dl, 'test' : test_dl}
    datasets_dict = {'train' : train_ds, 'val' : val_ds, 'test' : test_ds}
    
    return dataloaders_dict, datasets_dict


