# -*- coding: utf-8 -*-
"""
Train.py
---------------------------------------------------
File to train The netwrok on Data sets for solving
the restricted 3 body problem.
---------------------------------------------------
Created on Sun Aug 11 23:46:04 2024
@author: Andrew Francey
"""

import time
import torch
import os
import json, argparse
import pickle
import winsound
import torch.nn as nn
import source as src
import numpy as np
import pylab as plt


##--------------------------------------------
## Functions
##--------------------------------------------

def prep(folders, param):
    '''
    Process the data sets into workable tensors to train and test the model.
    Loads in the model, if one does not exist creates a new one.

    Parameters
    ----------
    folders : Listof(Int)
        The Numbers of the datasets to be used in training.
    param : JSON
        The hyperparameter for the data and the model.

    Returns
    -------
    data : Data 
        The class containing the training and testing data.
    model : Net
        The model to train.
    '''
    
    ## Load in the data to a working format.
    data = src.Data(folders, param['data'], device)
    
    ## If the model exists load in the model.
    if os.path.exists(args.model_name):
        with open(args.model_name) as model_file:
            model = pickle.load(model_file)
    else: # If it doesn't create a new one.
        model = src.Net(param['net'])
    
    model.to(device) # Move the model to the same device as the data.
    
    return data, model


def run(param, model, data):
    
    ## Using the ADAM optimization method for updating the models parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    
    ## Using the binary cross-entropy loss function to determine the loss value.
    loss = nn.BCELoss(reduction='mean')
    
    ## Create lists to store the testing and training loss values of each epoch.
    loss_vals = []
    cross_vals = []
    
    num_epochs = int(param['num_epochs'])
    for epoch in range(num_epochs):
        ## Passs the whole training data set through the model.
        train_val = model.backprop(data.values_train, data.targets_train, loss,
                                   optimizer)
        loss_val.append(train_val)
        
        ## Test the model on the test set.
        test_val = model.test(data.values_test, data.values_train, loss)
        cross_vals.append(test_val)
        
        ## Determine if the loss values should be printed to the screen.
        if epoch == 0:
            print('Epoch [{}/{}] ({:.1f}%)'.format(epoch+1, num_epochs, \
                                                   ((epoch+1)/num_epochs*100))+ \
                  '\tTraning loss: {:.5f}'.format(train_val) + \
                      'tTest loss: {:.5f}'.format(test_val))
                
        elif (epoch+1) % param['display_epochs'] == 0:
            print('Epoch [{}/{}] ({:.1f}%)'.format(epoch+1, num_epochs, \
                                                   ((epoch+1)/num_epochs*100))+ \
                  '\tTraning loss: {:.5f}'.format(train_val) + \
                      'tTest loss: {:.5f}'.format(test_val))
                winsound.Beep(1000,100)
        
    print('Final training Loss: {:.6f}'.format(loss_vals[-1]))
    print('Final test loss: {:.6f}'.format(cross_vals[-1]))        
    
    return loss_vals, cross_vals

##--------------------------------------------
## Main code
##--------------------------------------------

if __name__ == "__main__":
    
    start_time = time.time() # Get the start time
    
    ## Determine if a GPU is availible for use.
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    
    device = torch.device(dev)
    
    ## Determine the file path of file.
    file_location = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are needed for the training.
    parser = argparse.ArgumentParser(description="Training of 3-body solver")
    parser.add_argument('--param', default=file_location+'param.json',
                        type=str, help='Json file for hyperparameters.')
    parser.add_argument('--model-name', default=file_location+'3-body-solver.pkl',
                        type=str, help='Name of the model.')
    parser.add_argument('--fig-name', default=file_location+'loss_fig.png',
                        type=str, help='Name of the image to save teh loss plot as.')
    args = parser.parse_args()
    
    ## Open the hyperparameter file.
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    