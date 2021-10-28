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
from torchmetrics import Accuracy, MeanAbsolutePercentageError

from efficientnet_pytorch import EfficientNet
from torch import nn

on_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
import pickle


#%%

def train_model(model, dataloaders, criterion, metric, optimizer, num_epochs=25, is_inception=False, 
                regularization=None, classification=True):
    since = time.time()

    val_metric_history = []
    val_loss_history = []
    test_metric_history = []
    test_loss_history = []
    train_metric_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    # last_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            print("----- " + phase + "-----")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_metric = 0.0

            # Iterate over data.
            n_samples = 0
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                #print("\tbatch_idx = {}".format(batch_idx), end='')
                if not isinstance(criterion, nn.CrossEntropyLoss):  # If it is not the case of classification-problem
                    labels = labels.unsqueeze(dim=1).type(torch.FloatTensor)
                # labels = labels.unsqueeze(dim=1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    reg_loss = 0
                    reg_lambda = .001
                    if regularization is not None:
                        l1_penalty = torch.nn.L1Loss()
                        for param in model.parameters():
                            reg_loss += l1_penalty(param, torch.zeros(param.shape).to(device))           
                    # print('output shape: ', str(outputs.shape), ', label shape: ', str(labels.shape))
                    # print('output type: ', str(outputs.type()), ', label type: ', str(labels.type()))
                    data_loss = criterion(outputs, labels)
                    data_metric = metric(outputs, labels)   # batch metric mean
                    loss = data_loss + reg_lambda * reg_loss
                    print("\t\tdata_loss: {:.3f}, reg_loss: {:.3f}".format(data_loss, reg_lambda*reg_loss))

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_metric += data_metric * inputs.size(0)
                n_samples += len(inputs)
                if not on_cuda: break

            epoch_loss = running_loss / n_samples
            epoch_metric = running_metric / n_samples

            print('\t\t{}: {:.4f},  {}: {:.4f}\n'.format(str(criterion), epoch_loss, str(metric), epoch_metric))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print('-> Best validation loss!')
            # if phase == 'val' and epoch == num_epochs-1:
                # last_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_metric_history.append(epoch_metric)
                val_loss_history.append(epoch_loss)
            elif phase == 'test':
                test_metric_history.append(epoch_metric)
                test_loss_history.append(epoch_loss)
            elif phase == 'train':
                train_metric_history.append(epoch_metric)
                train_loss_history.append(epoch_loss)
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation {}: {:4f}'.format(str(criterion), best_loss))

    model_best = model
    model_best.load_state_dict(best_model_wts)
    # model_last = model
    # model_last.load_state_dict(last_model_wts)
       
    train_metric_history = [h.cpu().item() for h in train_metric_history]
    val_metric_history = [h.cpu().item() for h in val_metric_history]
    test_metric_history = [h.cpu().item() for h in test_metric_history]
    
    hist = [None]*2
    hist[0] = ((str(criterion)),(train_loss_history, val_loss_history, test_loss_history))
    hist[1] = ((str(metric)),(train_metric_history, val_metric_history, test_metric_history))
    
    models = (model_best, model_best)
    return models, hist


def eval_model(model, dataloader, score_fn, metric_fn, debug=False):
    
    # print('---- Evaluating the model -----')
    model.eval()   # Set model to evaluate mode
    n_samples = 0
    # batch_idx, (inputs, labels) = next(iter(enumerate(dataloader)))
    running_score = 0
    running_metric = 0
    # print("\t", end='')
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if debug and batch_idx >= 2:
            break
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        metric = metric_fn(outputs, labels)
        score = score_fn(outputs, labels)
        running_metric += metric * inputs.size(0)
        running_score += score * inputs.size(0)
        n_samples += len(inputs)
    # print()
    
    metric_final = running_metric / n_samples
    score_final = running_score / n_samples
    # print('\t{}: {:.2f}, {}: {:.2f}\n'.format(str(score_fn), score_final, str(metric_fn), metric_final))
    
    ###### AUC - ROS metric ######
    # running_labels_array = running_labels.numpy()
    # probs_array = (torch.softmax(running_outputs, dim=1)).detach().numpy()
    # fpr, tpr, thresholds = roc_curve(running_labels_array, probs_array[:,1])
    # plt.plot([0, 1], [0, 1], linestyle='--')
    # plt.plot(fpr, tpr)
    # plt.title('ROC curve'), plt.xlabel('False positive rate'), plt.ylabel('True positive rate'), plt.show()
    # auc_score = roc_auc_score(running_labels_array, probs_array[:,1])
    ##############################    
    
    return n_samples, score_final.item(), metric_final.item() #, auc_score


def eval_spearmanCorr(model, dataloader, debug=False):
    
    # print('\n---- Evaluating Spearman Rank Correlation -----')
    model.eval()   # Set model to evaluate mode
    n_samples = 0
    # dataloader = dataloaders_dict['val']
    # batch_idx, (inputs, targets) = next(iter(enumerate(dataloader)))
    running_outputs = torch.tensor([])
    running_outputs_prob = torch.tensor([])
    running_targets = torch.tensor([])
    softmax = nn.Softmax(dim=-1)
    # print("\t", end='')
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if debug and batch_idx >= 2:
            break
        inputs = inputs.to(device)
        # targets = targets.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs_prob = softmax(outputs)
        running_outputs = torch.cat((running_outputs, outputs.cpu()), dim=0)
        running_outputs_prob = torch.cat((running_outputs_prob, outputs_prob.cpu()), dim=0)
        running_targets = torch.cat((running_targets, targets), dim=0)
        n_samples += len(inputs)
    print()

    rho, pval = spearmanr(running_outputs[:,0].detach(), running_targets)
    rho_prob, pval_prob = spearmanr(running_outputs_prob[:,0].detach(), running_targets)

    # print('\tSpearman coefficient: {:.2f}, pval: {:.2f}\n'.format(rho, pval))
    # print('\tSpearman coefficient: {:.2f}, pval: {:.2f}\n'.format(rho_prob, pval_prob))
    
    return n_samples, rho_prob


def get_data_single_fold(model, dataloader, debug=False):
    
    model.eval()   # Set model to evaluate mode
    n_samples = 0
    # batch_idx, (inputs, labels) = next(iter(enumerate(dataloader)))
    running_outputs = torch.tensor([])
    running_outputs_prob = torch.tensor([])
    running_targets = torch.tensor([])
    running_labels = torch.tensor([])
    running_informations = []
    softmax = nn.Softmax(dim=-1)

    for batch_idx, (inputs, labels, targets, informations) in enumerate(dataloader):
        if debug and batch_idx >= 2:
            break
        labels = labels.long()
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs_prob = softmax(outputs)
        running_outputs = torch.cat((running_outputs, outputs.cpu()), dim=0)
        running_outputs_prob = torch.cat((running_outputs_prob, outputs_prob.cpu()), dim=0)
        running_labels = torch.cat((running_labels, labels.cpu()), dim=0)
        running_targets = torch.cat((running_targets, targets.cpu()), dim=0)
        running_informations += informations
        n_samples += len(inputs)
    
    return n_samples, np.array(running_outputs), np.array(running_outputs_prob), np.expand_dims(np.array(running_labels),axis=1), np.expand_dims(np.array(running_targets), axis=1), running_informations 


    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, classification=False, num_classes=1, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    if not classification:
        num_classes = 1

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
        input_size = 224

    elif model_name == "efficientnet-b1":
        """ Efficientnet-b1
        """
        model_ft = EfficientNet.from_pretrained(model_name)
        set_parameter_requires_grad(model_ft, feature_extract) #line to test
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 240

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


def plot_and_save(models, hist, out_folder, info_text):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
        with open(out_folder+'hist.pkl', 'wb') as file:
            pickle.dump(hist, file)
    
        (loss_name), (train_loss_history, val_loss_history, test_loss_history) = hist[0]
        (metric_name), (train_metric_history, val_metric_history, test_metric_history) = hist[1]
        
        fig, axs = plt.subplots(2, sharex=True)
        fig.suptitle('Training results')
        
        axs[0].set_title("Metric vs. Number of Training Epochs")
        axs[0].set(ylabel=metric_name)
        num_epochs = len(val_metric_history)
        axs[0].plot(range(1,num_epochs+1),train_metric_history,label="train metric")
        axs[0].plot(range(1,num_epochs+1),val_metric_history,label="val metric")
        axs[0].plot(range(1,num_epochs+1),test_metric_history,label="test metric")
        axs[0].grid(which='minor')
        axs[0].legend()
        axs[0].grid(b=True, which='both')
        
        axs[1].set_title("Loss vs. Number of Training Epochs")
        axs[1].set(xlabel='Training epochs', ylabel=loss_name)
        num_epochs = len(val_metric_history)
        axs[1].plot(range(1,num_epochs+1),train_loss_history,label="train loss")
        axs[1].plot(range(1,num_epochs+1),val_loss_history,label="val loss")
        axs[1].plot(range(1,num_epochs+1),test_loss_history,label="test loss")
        plt.xticks(np.arange(1, num_epochs+1, 1.0))
        axs[1].legend()
        axs[1].grid(b=True, which='both')
        
        fig.savefig(out_folder+'history.jpg', dpi=300)
        
        (model_last, model_best) = models
        # torch.save(model_last, out_folder+'model_last.pt')
        torch.save(model_best, out_folder+'model_best.pt')
        f = open(out_folder + "info.txt", "w")
        f.write(info_text)
        f.close()
    else:
        raise Exception('The output folder already exists: {}'.format(out_folder))
        
        
