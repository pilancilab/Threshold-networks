import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This script is just to create neural net models,
# In the code you encouter some nn model names, and we provide their corresponding names in the paper
# identity : Nonconvex-STE 
# ReLU     : Nonconvex-ReLU 
# LReLU    : Nonconvex-LReLU 
# BNN      : Nonconvex-CReLU 
# real_ReLU: ReLU (This does not show up often in the paper, but it is well-known relu network, withthreshold activation function) 



#############################################################
class STEFunction_identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class StraightThroughEstimator_identity(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_identity, self).__init__()

    def forward(self, x):
        x = STEFunction_identity.apply(x)
        return x
    
class NeuralNetwork_Unitstep_identity(torch.nn.Module):
    def __init__(self,nb_neuron,reduced_dim):
        super(NeuralNetwork_Unitstep_identity, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron,bias=False)
        self.actv = StraightThroughEstimator_identity()
        self.w2 = torch.nn.Linear(in_features=nb_neuron, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv(x)
        x = self.w2(x)
        return x





#############################################################
class STEFunction_identity_3layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class StraightThroughEstimator_identity_3layer(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_identity_3layer, self).__init__()

    def forward(self, x):
        x = STEFunction_identity_3layer.apply(x)
        return x
    
class NeuralNetwork_Unitstep_identity_3layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_Unitstep_identity_3layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1,bias=False)
        self.actv1 = StraightThroughEstimator_identity_3layer()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2,bias=False)
        self.actv2 = StraightThroughEstimator_identity_3layer()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        return x




#############################################################
class STEFunction_ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class StraightThroughEstimator_ReLU(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_ReLU, self).__init__()

    def forward(self, x):
        x = STEFunction_ReLU.apply(x)
        return x
    
class NeuralNetwork_Unitstep_ReLU(torch.nn.Module):
    def __init__(self,nb_neuron,reduced_dim):
        super(NeuralNetwork_Unitstep_ReLU, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron,bias=False)
        self.actv = StraightThroughEstimator_ReLU()
        self.w2 = torch.nn.Linear(in_features=nb_neuron, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv(x)
        x = self.w2(x)
        return x




#############################################################
class STEFunction_ReLU_3layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class StraightThroughEstimator_ReLU_3layer(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_ReLU_3layer, self).__init__()

    def forward(self, x):
        x = STEFunction_ReLU_3layer.apply(x)
        return x
    
class NeuralNetwork_Unitstep_ReLU_3layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_Unitstep_ReLU_3layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1,bias=False)
        self.actv1 = StraightThroughEstimator_ReLU_3layer()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2,bias=False)
        self.actv2 = StraightThroughEstimator_ReLU_3layer()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        return x




#############################################################
class STEFunction_LReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] *= 0.01 
        return grad_input

class StraightThroughEstimator_LReLU(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_LReLU, self).__init__()

    def forward(self, x):
        x = STEFunction_LReLU.apply(x)
        return x
    
class NeuralNetwork_Unitstep_LReLU(torch.nn.Module):
    def __init__(self,nb_neuron,reduced_dim):
        super(NeuralNetwork_Unitstep_LReLU, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron,bias=False)
        self.actv = StraightThroughEstimator_LReLU()
        self.w2 = torch.nn.Linear(in_features=nb_neuron, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv(x)
        x = self.w2(x)
        return x



#############################################################
class STEFunction_LReLU_3layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] *= 0.01 
        return grad_input

class StraightThroughEstimator_LReLU_3layer(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_LReLU_3layer, self).__init__()

    def forward(self, x):
        x = STEFunction_LReLU_3layer.apply(x)
        return x
    
class NeuralNetwork_Unitstep_LReLU_3layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_Unitstep_LReLU_3layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1,bias=False)
        self.actv1 = StraightThroughEstimator_LReLU_3layer()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2,bias=False)
        self.actv2 = StraightThroughEstimator_LReLU_3layer()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        return x



#############################################################
class STEFunction_BNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>1] = 0
        return grad_input


class StraightThroughEstimator_BNN(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_BNN, self).__init__()

    def forward(self, x):
        x = STEFunction_BNN.apply(x)
        return x
    
class NeuralNetwork_Unitstep_BNN(torch.nn.Module):
    def __init__(self,nb_neuron,reduced_dim):
        super(NeuralNetwork_Unitstep_BNN, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron,bias=False)
        self.actv = StraightThroughEstimator_BNN()
        self.w2 = torch.nn.Linear(in_features=nb_neuron, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv(x)
        x = self.w2(x)
        return x


#############################################################
class STEFunction_BNN_3layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        return (output > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[input>1] = 0
        return grad_input

class StraightThroughEstimator_BNN_3layer(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator_BNN_3layer, self).__init__()

    def forward(self, x):
        x = STEFunction_BNN_3layer.apply(x)
        return x
    
class NeuralNetwork_Unitstep_BNN_3layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_Unitstep_BNN_3layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1,bias=False)
        self.actv1 = StraightThroughEstimator_BNN_3layer()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2,bias=False)
        self.actv2 = StraightThroughEstimator_BNN_3layer()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        return x



#############################################################
class NeuralNetwork_real_ReLU(torch.nn.Module):
    def __init__(self,nb_neuron,reduced_dim):
        super(NeuralNetwork_real_ReLU, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron,bias=False)
        self.actv = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(in_features=nb_neuron, out_features=1,bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.actv(x)
        x = self.w2(x)
        return x

class NeuralNetwork_real_ReLU_3layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_real_ReLU_3layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1,bias=False)
        self.actv1 = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2,bias=False)
        self.actv2 = torch.nn.ReLU()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=1,bias=False)
    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        return x



class NeuralNetwork_real_ReLU_5layer(torch.nn.Module):
    def __init__(self,nb_neuron1,nb_neuron2,reduced_dim):
        super(NeuralNetwork_real_ReLU_5layer, self).__init__()
        self.w1 = torch.nn.Linear(in_features=reduced_dim, out_features=nb_neuron1)
        self.actv1 = torch.nn.ReLU()
        self.w2 = torch.nn.Linear(in_features=nb_neuron1, out_features=nb_neuron2)
        self.actv2 = torch.nn.ReLU()
        self.w3 = torch.nn.Linear(in_features=nb_neuron2, out_features=nb_neuron2)
        self.actv3 = torch.nn.ReLU()
        self.w4 = torch.nn.Linear(in_features=nb_neuron2, out_features=nb_neuron2)
        self.actv4 = torch.nn.ReLU()
        self.w5 = torch.nn.Linear(in_features=nb_neuron2, out_features=1)
    def forward(self, x):
        x = self.w1(x)
        x = self.actv1(x)
        x = self.w2(x)
        x = self.actv2(x)
        x = self.w3(x)
        x = self.actv3(x)
        x = self.w4(x)
        x = self.actv4(x)
        x = self.w5(x)
        return x


