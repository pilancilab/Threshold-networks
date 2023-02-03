from modelsnew import *
import torch 
import torch.optim as optim
import numpy as np
import numpy.linalg as la
from torch.optim.lr_scheduler import ExponentialLR
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time 


class train:
    def __init__(self,nn_model,X,Y,Xtest,Ytest,beta,n_train,n_test,nb_neuron,nb_neuron2,nb_epoch,reduced_dim,lr):
        self.nn_model = nn_model
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.beta = beta
        self.n_train = n_train
        self.n_test = n_test
        self.nb_neuron = nb_neuron
        self.nb_neuron2 = nb_neuron2
        self.nb_epoch = nb_epoch
        self.reduced_dim = reduced_dim
        self.loss_values = []
        self.testacc_time = []
        self.trainacc_time = []
        self.objval_time = []
        self.time = []
        self.lr = lr
    
    ###creating a neural network model #####
    def training(self):
        if self.nn_model == "identity_2":
            Network = NeuralNetwork_Unitstep_identity(self.nb_neuron,self.reduced_dim)
        if self.nn_model == "ReLU_2":
            Network = NeuralNetwork_Unitstep_ReLU(self.nb_neuron,self.reduced_dim)
        if self.nn_model == "LReLU_2":
            Network = NeuralNetwork_Unitstep_LReLU(self.nb_neuron,self.reduced_dim)
        if self.nn_model == "BNN_2":
            Network = NeuralNetwork_Unitstep_BNN(self.nb_neuron,self.reduced_dim)
        if self.nn_model =="real_ReLU_2":
            Network = NeuralNetwork_real_ReLU(self.nb_neuron,self.reduced_dim)
        if self.nn_model == "identity_3":
            Network = NeuralNetwork_Unitstep_identity_3layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)
        if self.nn_model == "ReLU_3":
            Network = NeuralNetwork_Unitstep_ReLU_3layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)
        if self.nn_model == "LReLU_3":
            Network = NeuralNetwork_Unitstep_LReLU_3layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)
        if self.nn_model == "BNN_3":
            Network = NeuralNetwork_Unitstep_BNN_3layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)
        if self.nn_model =="real_ReLU_3":
            Network = NeuralNetwork_real_ReLU_3layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)
        if self.nn_model =="real_ReLU_5":
            Network = NeuralNetwork_real_ReLU_5layer(self.nb_neuron,self.nb_neuron2,self.reduced_dim)

        # if torch.cuda.is_available():
        #     device = "cuda"
        # else: 
        #     device = "cpu"
        #     print("CUDA IS NOT AVAILABLE")
        
        device = "cpu"
        ###### From numpy arrays to torch tensors #####
        X_ten = torch.tensor(self.X, dtype=torch.float).to(device)
        Y_ten = torch.tensor(self.Y, dtype=torch.float).to(device)
        Y_ten = Y_ten.reshape(shape=(self.n_train,1)).to(device)
        Y_ten_with_zeros = Y_ten.detach().clone().to(device)

        optimizer = optim.SGD(params=Network.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss(reduction="mean") 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,eps=1e-12)
        X_test_ten = torch.tensor(self.Xtest, dtype=torch.float).to(device)
        Y_test_ten = torch.tensor(self.Ytest, dtype=torch.float).to(device)
        Y_test_ten = Y_test_ten.reshape(shape=(self.n_test,)).to(device)

        Network.to(device)

        ###### Training Process   ######
        for _ in range(int(self.nb_epoch)):

            start_time = time.time()
            optimizer.zero_grad()
            out = Network(X_ten)
            loss = criterion(out,Y_ten_with_zeros)/(2)
            loss += self.beta * (torch.norm(Network.w2.weight, p=2) + torch.norm(Network.w1.weight, p=2)) / (2*self.n_train)
            self.loss_values.append(loss.data)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            self.time.append(end_time-start_time)    #### For each iteration we append the required time to the time list 
            scheduler.step(loss)


            output = Network(X_test_ten)
            output = torch.reshape(output, shape=(self.n_test,))
            output[output < 0] = -1
            output[output >= 0] = 1
            nn_acc = int((output == Y_test_ten).sum().data) /self.n_test
            self.testacc_time.append(nn_acc)           ##### For each iteration we also keep the test accuracies, and append them into a list


            output = Network(X_ten)
            output = torch.reshape(output, shape=(self.n_train,))
            output[output < 0] = -1
            output[output >= 0] = 1
            Y_ten = Y_ten.reshape(shape=(self.n_train,))
            nn_acc = int((output == Y_ten).sum().data) /self.n_train
            self.trainacc_time.append(nn_acc)      ##### For each iteration we also keep the train accuracies, and append them into a list

        
        for i in range(1,len(self.time)):
            self.time[i] += self.time[i-1]


        return np.array(self.loss_values),self.testacc_time,self.trainacc_time,self.time

