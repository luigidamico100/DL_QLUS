#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:54:22 2021

@author: luigidamico
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
https://github.com/lukemelas/EfficientNet-PyTorch
"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from efficientnet_pytorch import EfficientNet
from torch import nn

on_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
data_dir = "./data/hymenoptera_data"


#%%

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, regularization=None):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
        
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        reg_loss = 0
                        reg_lambda = .001
                        if regularization is not None:
                            l1_penalty = torch.nn.L1Loss()
                            for param in model.parameters():
                                reg_loss += l1_penalty(param, torch.zeros(param.shape).to(device))            
                        data_loss = criterion(outputs, labels)
                        loss = data_loss + reg_lambda * reg_loss
                        print("\t\tdata_loss: {:.3f}, reg_loss: {:.3f}".format(data_loss, reg_loss))

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if not on_cuda:
                    break
            print()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch == num_epochs-1:
                last_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            elif phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model_best = model
    model_best.load_state_dict(best_model_wts)
    model_last = model
    model_last.load_state_dict(last_model_wts)
       
    train_acc_history = [h.cpu().item() for h in train_acc_history]
    val_acc_history = [h.cpu().item() for h in val_acc_history]
    hist = (train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    models = (model_last, model_best)
    return models, hist


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "efficientnet-b0":
        """ Efficientnet-b0
        """
        model_ft = EfficientNet.from_pretrained(model_name)
        set_parameter_requires_grad(model_ft, feature_extract) #line to test
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 0                      # to define

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft.to(device), input_size


# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         #the models were pretrained with the hard-coded normalization values, as described here 
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }


def get_params_to_update(model, feature_extract):
    #If feature_extract==True usually mean that the params to learn are few.
    #Otherwise, for fine-tuning training the params are many, so a generator 
    #is returned in place of the parameters list 
    if feature_extract:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update
        

def print_model_parameters(model):
    print('************************************** Model parameters summary ***************************')
    for name, parameter in model.named_parameters():
        print(name + "\t\t\t\t\t" + parameter.shape.__str__() + "\t\t ReqGrad? " + parameter.requires_grad.__str__())
    
    print('************************************** Model total parameters summary ***************************')
    total_params = sum(p.numel() for p in model.parameters())
    total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: %d' %total_params)
    print('Total learnable parameters: %d' %total_learnable_params)
    
    
def plot_and_save(models, hist, out_folder):
    (train_loss_history, train_acc_history, val_loss_history, val_acc_history) = hist
    
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    num_epochs = len(val_acc_history)
    plt.plot(range(1,num_epochs+1),train_acc_history,label="test acc")
    plt.plot(range(1,num_epochs+1),val_acc_history,label="val acc")
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(out_folder+'Accuracy.jpg')
    plt.show()
    
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    num_epochs = len(val_acc_history)
    plt.plot(range(1,num_epochs+1),train_loss_history,label="test loss")
    plt.plot(range(1,num_epochs+1),val_loss_history,label="val loss")
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig(out_folder+'Loss.jpg')        
    plt.show()
    
    (model_last, model_best) = models
    torch.save(model_last, out_folder+'model_last.pt')
    torch.save(model_best, out_folder+'model_best.pt')



