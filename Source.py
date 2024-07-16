# -*- coding: utf-8 -*-
"""
3-body ANN solver
Source file
----------------------------
Author: Andrew Francey
----------------------------
Data:12/07/24
"""

import os
import torch
import pickle as pck
import numpy as np
import torch.nn as nn
import torch.nn.functional as func


##-----------------------------------------------------------------------------
## Class's
##-----------------------------------------------------------------------------

class Data():
    
    def __init__(self, folders, param, device):
        
        data_path = os.getcwd()+'\\Datasets\\Dataset'
        
        for i in folders:
            folder = data_path+str(i)
            
            if not(os.path.exists(folder)):
                raise Exception("The folder Dataset{} does not exists".format(i))
        
        targets = []
        values = []
        
        for i in folders:
            folder = data_path+str(i)
            
            with open(folder+'\\Time.pkl', 'r') as f:
                time_param = pck.load(f)
            f.close()
            
            with open(folder+'\\Values.pkl', 'r') as f:
                vals = pck.load(f)
            f.close()
            
            with open(folder+'\\Targets.pkl', 'r') as f:
                targ = pck.load(f)
            f.close()
            
            T = time_param[0]
            dt = time_param[1]
            
            n_vals = np.shape(vals)[1]
            t = np.arange(0, T+dt, dt)
            
            value = np.zeros((n_vals*t.size, 5),float)
            
            for n in range(n_vals):
                for m in range(t.size):
                    value[n*t.size + m][0] = t[m]
                    value[n*t.size + m][1] = vals[n][0][0]
                    value[n*t.size + m][2] = vals[n][0][1]
                    value[n*t.size + m][3] = vals[n][1][0]
                    value[n*t.size + m][4] = vals[n][1][1]
            
            values.append(value)
            
            target = np.zeros((n_vals*t.size,2),float)
            for n in range(n_vals):
                for m in range(t.size):
                    target[n*t.size + m][0] = targ[n][0]
                    target[n*t.size + m][1] = targ[n][1]
            
            targets.append(target)
        
        values = values.reshape(-1,5)
        targets = targets.reshape(-1,2)
        
        self.values = torch.tensor(values).to(device)
        self.targets = torch.tensor(targets).to(device)
            
            
        

class Net(nn.Module):
    '''
    A Artifical neural network with 3 hidden layer designed to approximate the
    solution to a 2nd order differental equaiton with 5 inputs and 2 outputs. 
    Using Tanh activation functions on the hidden layers and a linear 
    activation function on the output layer.
    
    Uses the torch.nn.Module as a parent class.
    
    Parameters
    ----------
    net_params : JSON file
        Json file with the networks hyperparameters.
        
    Attributes
    ----------
    layer1 : Sequential
        A fully connected linear layer with a Tanh activation.
    layer2 : Sequential
        A fully connected linear layer with a Tanh activation.
    layer3 : Sequential
        A fully connected linear layer with a Tanh activation.
    Layer4 : Sequential
        A fully connected linear layer with a Linear activation.
    
    Functions
    ---------
    forward(self, x)
        Runs a data group x throught the network outputting a tensor with the
        prediction of the network.
    backprop(self, inputs, targets, loss, optimizer)
        Runs a data group inputs through the network comparing to the target 
        group targets using the loss function loss. Then using an optimizer to
        update the weights of the network.
    test(self, data, loss)
        Runs a data group through the network and comparing its data and 
        targets using a loss function. test does not update any weights.
    '''
    
    def __init__(self, net_params):
        super(Net, self).__init__()
        
        ## Retreving the network hyperparameters from the json file.
        hidden1 = net_params['hidden_1']
        hidden2 = net_params['hidden_2']
        hidden3 = net_params['hidden_3']
        drop_out_rate = net_params['drop_out']
        
        ## Create the network.
        self.layer1 = nn.Linear(5,hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, hidden3)
        self.layer4 = nn.Linear(hidden3, 2)
        
        self.drop_out = nn.Dropout(drop_out_rate)

        
    def forward(self, x):
        '''
        Passes a torch tensor x of the data through the models network and 
        returns a torch tensro with the models guesses.

        Parameters
        ----------
        x : Torch Tensor
            Data to be evaluated by the model.

        Returns
        -------
        Torch Tensor
            The models best guess at the solution.
        '''
        
        x = self.layer1(x)
        x = func.tanh(x)
        x = self.drop_out(x)
        x = self.layer2(x)
        x = func.tanh(x)
        x = self.drop_out(x)
        x = self.layer3(x)
        x = func.tanh(x)
        x = self.drop_out(x)
        x = self.layer4(x)
        x = func.Linear(x)
        
        return x

    
    def backprop(self, inputs, targets, loss, optimizer):
        '''
        Passes a data set inputs through the model then using a loss function
        loss to compare the models output to the target targets. Then update the
        weights of the model using the optimizer.

        Parameters
        ----------
        inputs : Torch Tensor
            The data set to be evaluated by the model.
        targets : Torch Tensor
            The data sets target.
        loss : Torch Loss Function
            The loss function to use in determining the loss value.
        optimizer : Torch Optimizer
            The optimization function to determine how to update the weights.

        Returns
        -------
        Float
            The loss value of the training.
        '''
        
        self.train()  # Set the model to train mode.
        
        outputs = self.forward(inputs)   # Pass the data through the model.
        obj_val = loss(outputs, targets) # Compute the loss value.
        optimizer.zero_grad()            # Resent the gradient.
        obj_val.backward()               # Update the models values.
        optimizer.step()                 # Inrement the step counter.
        
        return obj_val.item()

    
    def test(self, inputs, targets, loss):
        '''
        Passes the data set through the model and returning the loss value 
        using the loss function to evaluate the datas inputs and targets.

        Parameters
        ----------
        data : Data object
            The data set.
        loss : Torch Loss Function
            The loss function to use in determining the loss value.

        Returns
        -------
        Float
            The loss value of the testing.
        '''
        
        self.eval()   # Set the model to eval mode.
        
        with torch.no_grad():  # Evalute the model without using gradients.
            outputs = self.forward(inputs)   # Pass the data through the model.
            cross_val = loss(outputs, targets) # Compute loss value.
        
        return cross_val.item()
    
    
        
