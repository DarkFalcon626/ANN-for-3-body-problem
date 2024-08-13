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
import Source as src
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
        model_file.close()
    else: # If it doesn't create a new one.
        model = src.Net(param['net'])
    
    model.to(device) # Move the model to the same device as the data.
    
    return data, model


def run(param, model, data):
    '''
    Train and test the model outputing the loss values from both the training
    and testing at each epoch. 

    Parameters
    ----------
    param : JSON file
        Hyperparameters for the training of the model. Such as the learning
        rate, number of epochs, the interval of epochs to display.
    model : Net
        The ANN network to train and test.
    data : Data
        The data set for testing and training of the model.

    Returns
    -------
    loss_vals : listof (Floats)
        The loss values per epoch of training.
    cross_vals : listof (Floats)
        The loss values per epoch of testing.
    '''
    
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
        loss_vals.append(train_val)
        
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


def save_value(save):
    '''
    Determines if the input is a excepted value to a yes or no question, if not
    get a new input.

    Parameters
    ----------
    save : STR
        Input string to a yes or no question.

    Returns
    -------
    save : Bool
        Awnser to the yes or no question.
    '''
    
    ## Reduce any uppercase to lowercase to maintain the meaning of the word.
    save = save.lower()
    
    ## Determine if the awnser is true or false.
    if save in ['true', '1', 'yes']:
        save = True
    elif save in ['false', '0', 'no']:
        save = False
    else: # If the awnser is not an excepted value ask the question again.
        save = input('Value entered is not a proper response. Please enter \
                     either true, 1, yes or false, 0, no. ->')
        save = save_value(save) # Check the new awnser.
    
    return save


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
    
    ## Take the datasets to train the model on.
    input_string = input("Enter numbers of folders of datasets to train model on (i.e 0 1 4): ")
    
    inputs = input_string.split() # Split into list of the numbers.
    
    ## Convert the string to integers.
    folders = []
    for i in inputs:
        folders.append(int(i))
    
    print("Processing Data...")
    data, model = prep(folders, param) # Process the data and create/load the model.
    
    print('Data processed')
    print('Beginning training...')
    loss_vals, cross_vals = run(param['exec'], model, data) # Train the model.
    
    train_time = time.time()-start_time # Determine the time took to train.
    ## Convert the time into a more readable formate of hours, minutes and seconds.
    if (train_time//3600) > 0:
        hours = train_time//3600
        mins = (train_time-hours*3600)//60
        secs = train_time-hours*3600-mins*60
        print('The model trained in {} hours, {} mins and {:.0f} seconds'.format(hours,mins,secs))
    elif (train_time//60) > 0:
        print('The model trained in {} mins and {:.0f} seconds'.format(train_time//60,
                                                               train_time-(train_time//60)*60))
    else:
        print('The model trained in {:.0f} seconds'.format(train_time))
    
    x = np.arange(1,len(loss_vals)+1) # Axis for the number of epochs.  
    
    ## Plot the loss values vs the epoch.
    plt.plot(x,loss_vals, label='Training loss')
    plt.plot(x,cross_vals, label='Test loss')
    plt.title('Loss per Epoch.')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    ## Ask if the model and figures should be saved.
    save = input('Do you want to save the model and training data: ')
    save = save_value(save)
    
    if save:
        with open(args.model_name) as model_file:
            pickle.dump(model, model_file)
        model_file.close()
        plt.savefig(args.fig_name, format='png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    