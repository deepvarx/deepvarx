# def getLinVARX(name, exoInput=None):
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite import metrics
from absl import app, flags

class LinVARX(nn.Module):

    def __init__(self, lags):
        super(LinVARX, self).__init__()
        self.lags = lags 
        self.endoDense = nn.Linear(self.lags * 2, 2)
        self.exoDense = nn.Linear(self.lags * 66, 2)
        self.flatten = nn.Flatten()
        self.optimiser = optim.SGD(self.parameters(), lr=1e-7, momentum=0.9, weight_decay=1e-3)
        self.loss_fn = nn.MSELoss()
        self.metrics = metrics.MeanSquaredError()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        endo, exo = inputs 
        endo = self.flatten(endo)
        exo = self.flatten(exo)
        endoDense = self.endoDense(endo)
        exoDense = self.exoDense(exo)

        middle = endoDense + exoDense

        return self.sigmoid(middle)

class DeepVARX(nn.Module):

    def __init__(self, lags, exoInput=1, gazeOnly=False):
        
        super(DeepVARX, self).__init__()
        self.gazeOnly = gazeOnly        
        self.lags = lags
        self.endoConv1 = nn.Conv1d(2, 16, kernel_size=15)
        self.endoPool1 = torch.nn.MaxPool1d(2)
        self.endoBatchNorm1 = nn.BatchNorm1d(16)
        self.endoActivation1 = nn.ReLU()

        self.endoConv2 = nn.Conv1d(16, 32, kernel_size=10)
        self.endoPool2 = torch.nn.MaxPool1d(2)
        self.endoBatchNorm2 = nn.BatchNorm1d(32)
        self.endoActivation2 = nn.ReLU()
        
        self.endoConv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.endoBatchNorm3 = nn.BatchNorm1d(64)
        self.endoActivation3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
        if not gazeOnly:

            self.exoConv1 = nn.Conv1d(exoInput, 32, kernel_size=15)
            self.exoPool1 = torch.nn.MaxPool1d(2)
            self.exoBatchNorm1 = nn.BatchNorm1d(32)
            self.exoActivation1 = nn.ReLU()

            self.exoConv2 = nn.Conv1d(32, 64, kernel_size=10)
            self.exoPool2 = torch.nn.MaxPool1d(2)
            self.exoBatchNorm2 = nn.BatchNorm1d(64)
            self.exoActivation2 = nn.ReLU()
            
            self.exoConv3 = nn.Conv1d(64, 128, kernel_size=5)
            self.exoBatchNorm3 = nn.BatchNorm1d(128)
            self.exoActivation3 = nn.ReLU()

            self.exoPool4 = torch.nn.MaxPool1d(2)
            self.sigmoid = nn.Sigmoid()
            
        self.flatten = nn.Flatten()
         
        outShape = self.lags
        for (kernel_size, stride) in zip([15, 2, 10, 2, 5], [1, 2, 1, 2, 1]):
            outShape = self.getConvSize(in_shape=outShape, 
                                        padding=0, 
                                        dilation=1,
                                        kernel_size=kernel_size,
                                        stride=stride
                                        )
        
        self.outDense1 = nn.Linear(int(outShape)*64, 64)
        self.outActivation3 = nn.ReLU()
        self.outDense2 = nn.Linear(64, 2)
        self.outActivation4 = nn.ReLU()

        self.optimiser = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        self.metrics = metrics.MeanSquaredError()
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def getConvSize(in_shape, padding, dilation, kernel_size, stride):
        return ((in_shape + 2*padding - dilation*(kernel_size-1)-1)/stride)+1

    def forward(self, inputs):
        if self.gazeOnly:
            endo = inputs.permute(0, 2, 1)
        else:
            endo, exo = inputs 
            endo = endo.permute(0, 2, 1)
            exo = exo.permute(0, 2, 1)

        endo = self.endoConv1(endo)
        endo = self.endoBatchNorm1(endo) 
        endo = self.endoActivation1(endo)
        endo = self.endoPool1(endo)

        endo = self.endoConv2(endo)
        endo = self.endoBatchNorm2(endo) 
        endo = self.endoActivation2(endo)
        endo = self.endoPool2(endo)
        
        endo = self.endoConv3(endo)
        endo = self.endoBatchNorm3(endo) 
        endo = self.endoActivation3(endo)

        if self.gazeOnly:
            middle = endo
        else:
            exo = self.exoConv1(exo)
            exo = self.exoBatchNorm1(exo)
            exo = self.exoActivation1(exo)
            exo = self.exoPool1(exo)

            exo = self.exoConv2(exo)
            exo = self.exoBatchNorm2(exo)
            exo = self.exoActivation2(exo)
            exo = self.exoPool2(exo)
            
            exo = self.exoConv3(exo)
            exo = self.exoBatchNorm3(exo)
            exo = self.exoActivation3(exo)
            
            exo = exo.permute(0, 2, 1)
            exo = self.exoPool4(exo)
            exo = exo.permute(0, 2, 1)
            middle = endo + exo

        middle = self.flatten(middle)

        middle = self.outDense1(middle)
        middle = self.outActivation3(middle)
        return self.sigmoid(self.outDense2(middle))
