# -*- coding: utf-8 -*-
"""
3-body ANN solver
Source file
----------------------------
Author: Andrew Francey
----------------------------
Date:12/07/24
"""

import os
import torch
import json
import pickle as pck
import numpy as np
import torch.nn as nn
import torch.nn.functional as func


##--------------------------------------------------------
## Class's
##--------------------------------------------------------

class Data():
    '''
    Consumes a list of numbers corresponding to folders for datasets, a
    JSON file with parameters for the dataset, and the device either a GPU
    or the CPU to store the data on. This creates a dataset with attributes
    for testing and training including input values and targets to train a 
    ANN netwrok.

    Parameters
    ----------
    folders : Listof(Int)
        A list of the indices corresponding to folders for data sets.
    param : JSON 
        The parameters for how to set up the datasets.
    device : Torch Object
        Either the CPU or the GPU to store the data on.
    random : Bool
        Boolean value telling if the data should be shuffled randomly if True,
        or stay in organized structures if False. Default is False.

    Raises
    ------
    Exception
        If file doesn't exist.
    
    Attributes
    ----------
    n_test : Int
        The number of data points in the test set.
    n_train : Int
        The number of data points in the training set.
    
    '''
    def __init__(self, folders, param, device):
        
        ## Assign the location of the datasets to a datapath.
        data_path = os.path.dirname(__file__)+'\\Datasets\\Dataset'
        
        ## The following ensures that all the numbers in folders corresponds to 
        ## an existing data set.
        for i in folders:
            folder = data_path+str(i)
            
            ## If the folder does not exist raise an error.
            if not(os.path.exists(folder)):
                raise Exception("The folder Dataset{} does not exists".format(i))
        
        ## Initial lists to store the targets and the values.
        targets = []
        values = []
        
        ## Iterate through all the folders.
        for i in folders:
            folder = data_path+str(i) #Get the path for the folder.
            
            ## Open the files in the folder.
            with open(folder+'\\Time.pkl', 'rb') as f:
                time_param = pck.load(f)
            f.close()
            
            with open(folder+'\\Values.pkl', 'rb') as f:
                vals = pck.load(f)
            f.close()
            
            with open(folder+'\\Targets.pkl', 'rb') as f:
                targ = pck.load(f)
            f.close()
            
            ## Load in the parameters
            T = time_param[0]
            dt = time_param[1]
            n = time_param[2]
            
            ## Create an array for the time parameters.
            t = np.arange(0, T+dt, dt)

            ## Create an (1,5) vector to feed into the network.
            for i in range(n):
                val_set = np.zeros((t.size,5),float)
                for j in range(t.size):
                    val_set[j][0] = vals[0][i][0]
                    val_set[j][1] = vals[0][i][1]
                    val_set[j][2] = vals[1][i][0]
                    val_set[j][3] = vals[1][i][1]
                    val_set[j][4] = t[j]
                
                ## Appending our sets of vectors to the list of sets.
                values.append(val_set)
                targets.append(targ[i])
        
        ## Determine the sizes for the test and training datasets.
        test_size = round(len(targets)*param['test_percentage'])
        train_size = len(targets) - test_size
        
        ## Convert to arrays
        targets_train = np.array(targets[:train_size])
        targets_test = np.array(targets[train_size:])
        
        values_train = np.array(values[:train_size])
        values_test = np.array(values[train_size:])

        ## Reshape the arrays
        targets_train = targets_train.reshape((targets_train.shape[0]*targets_train.shape[1], targets_train.shape[2]))
        targets_test = targets_test.reshape((targets_test.shape[0]*targets_test.shape[1], targets_test.shape[2]))
        
        values_train = values_train.reshape((values_train.shape[0]*values_train.shape[1], values_train.shape[2]))
        values_test = values_test.reshape((values_test.shape[0]*values_test.shape[1], values_test.shape[2]))
        
        ## Get the lenght of the test and training data.
        self.n_train = targets_train.shape[0]
        self.n_test = targets_test.shape[0]
        
        ## Convert to pytorch tensors.
        self.targets_train = torch.tensor(targets_train).to(device)
        self.targets_test = torch.tensor(targets_test).to(device)
        
        self.values_train = torch.tensor(values_train).to(device)
        self.values_test = torch.tensor(values_test).to(device)
    
    def shuffle(self, device=torch.device('cpu')):
        '''
        Shuffles up the data for the training and the test data sets keeping
        the targets and the values matching in the indices

        Returns
        -------
        None.

        '''
        
        ## Create an array with all the indices of the training set.
        indexs = np.arange(0, self.n_train, 1)
            
        ## Shuffle the indices.
        np.random.shuffle(indexs)
            
        indexs = torch.tensor(indexs).to(device)
        
        self.targets_train = torch.index_select(self.targets_train,0,indexs)
        self.values_train = torch.index_select(self.values_train,0,indexs)
        
        indexs = np.arange(self.n_test)
        
        np.random.shuffle(indexs)
        
        indexs = torch.tensor(indexs).to(device)
        
        self.targets_test = torch.index_select(self.targets_test, 0, indexs)
        self.values_test = torch.index_select(self.values_test, 0, indexs)
    

    def create_batch(self, n_batches, shuffle = True, device=torch.device('cpu')):
        '''
        Reshapes the data into batchs for training. If data can not be broken
        into equal sized batchs, last batch will be dropped.

        Parameters
        ----------
        n_batches : Int
            The number of batchs to split the data into.
        shuffle : Bool, optional
            If true the data will be shuffled. The default is True.

        Returns
        -------
        targets_train : Torch Tensor
            The training targets batched data.
        values_train : Torch Tensor
            The training values batched data.
        '''
        
        if shuffle:
            self.shuffle(device)

        batch_size = self.n_train//n_batches
        
        rem = self.n_train%batch_size
            
        targets_train = self.targets_train[:-rem]
        values_train = self.values_train[:-rem]
        
        targets_train = torch.reshape(targets_train,(n_batches,
                                                     batch_size,
                                                     targets_train.shape[1]))
        values_train = torch.reshape(values_train, (n_batches,
                                                    batch_size,
                                                    values_train.shape[1]))
        
        return targets_train, values_train
        
        

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
        hidden4 = net_params['hidden_4']
        hidden5 = net_params['hidden_5']
        drop_out_rate = net_params['drop_out']
        
        ## Create the network.
        self.fc1 = nn.Sequential(nn.Linear(5, hidden1),
                                 nn.Tanh(),
                                 nn.BatchNorm1d(hidden1))
        self.fc2 = nn.Sequential(nn.Linear(hidden1, hidden2),
                                 nn.Tanh(),
                                 nn.BatchNorm1d(hidden2))
        self.fc3 = nn.Sequential(nn.Linear(hidden2, hidden3),
                                 nn.Tanh(),
                                 nn.BatchNorm1d(hidden3))
        self.fc4 = nn.Sequential(nn.Linear(hidden3, hidden4),
                                 nn.Tanh(),
                                 nn.BatchNorm1d(hidden4))
        self.fc5 = nn.Sequential(nn.Linear(hidden4, hidden5),
                                 nn.Tanh(),
                                 nn.BatchNorm1d(hidden5))
        self.fc6 = nn.Linear(hidden5, 2)
        
        self.drop_out = nn.Dropout(drop_out_rate)
        
        self.double()

        
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
        
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        x = self.fc3(x)
        x = self.drop_out(x)
        x = self.fc4(x)
        x = self.drop_out(x)
        x = self.fc5(x)
        x = self.drop_out(x)
        x = self.fc6(x)

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
    
##-----------------------------------------------------------------------------
## Main Code For testing
##-----------------------------------------------------------------------------   
        
## The following section of code is design to be used to test that all the 
##  class and functions run smoothly.

if __name__ == '__main__':
    
    ## Determine if a GPU is availabe to train.
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
        
    device = torch.device(dev)
    
    ## Determines the location of the file with the parameters.
    param_location = os.path.dirname(__file__) + '\\param.json'
    
    ## Open the parameter file.
    with open(param_location, 'rb') as f:
        param = json.load(f)
    
    f.close
    
    ## Create a data and network class.
    data = Data([0,1], param['data'], device)
    model = Net(param['net']).to(device)
    x = model.forward(data.values_train)
    print('Data and model created successfully.')
    data.shuffle(device)
    print('Shuffled')
    data_targets, data_values = data.create_batch(5, False, device)