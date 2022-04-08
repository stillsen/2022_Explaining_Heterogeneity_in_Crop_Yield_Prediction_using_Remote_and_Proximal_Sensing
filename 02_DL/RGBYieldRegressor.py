# -*- coding: utf-8 -*-
"""
Crop yield regression model implementing a ResNet50 in PyTorch Lightning in a transfer learning approach using
RGB remote sensing observations
"""

# Built-in/Generic Imports

# Libs
import torch
import torchvision.models as models
import torch.nn as nn
from torch.optim import SGD, Adam

from pytorch_lightning.core.lightning import LightningModule

# Own modules


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

class RGBYieldRegressor(LightningModule):
    def __init__(self, optimizer:str = 'sgd', k:int = 0, lr:float = 0.001, momentum:float = 0.8, wd:float = 0.01, batch_size:int = 16, pretrained:bool = True, tune_fc_only:bool = False, model: str = 'resnet50'):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.batch_size = batch_size
        self.k = k

        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]

        self.criterion = nn.MSELoss(reduction='mean')

        self.model_arch = model

        num_target_classes = 1
        if self.model_arch == 'resnet50':
            print('setting up resnet with lr = {}, m = {}, wd = {}'.format(lr, momentum, wd))
            # init a pretrained resnet
            self.model = models.resnet50(pretrained=pretrained)
            num_filters = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.ReLU(),
                nn.Linear(num_filters, num_target_classes))

        elif self.model_arch == 'densenet':
            print('setting up densenet with lr = {}, m = {}, wd = {}'.format(lr, momentum, wd))
            self.model = models.densenet121(pretrained=pretrained)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(num_filters, num_target_classes))

        if pretrained:
            if tune_fc_only: # option to only tune the fully-connected layers
                for child in list(self.model.children())[:-1]:
                    for param in child.parameters():
                        param.requires_grad = False

    def forward(self, x):
        # self.feature_extractor.eval() # set model in evaluation mode -> DOESN'T Lightning do this automatically?
        # with torch.no_grad():
            # representations = self.feature_extractor(x).flatten(1)
        # return self.regressor(representations)
        return torch.flatten(self.model(x))
        # return self.model(x)

    def training_step(self, batch, batch_idx): # torch.autograd?
        x, y = batch
        y_hat = torch.flatten(self.model(x))
        loss = self.criterion(y_hat, y)
        # self.log("train_loss_{}".format(self.k), loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        # return loss_fn(torch.flatten(y_hat), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self.model(x))
        loss = self.criterion(y_hat, y)
        # perform logging
        # self.log("val_loss_{}".format(self.k), loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.flatten(self.model(x))
        loss = self.criterion(y_hat, y)
        # perform logging
        # self.log("test_loss_{}".format(self.k), loss, prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def predicts_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model(batch).squeeze()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)


    #def prediction_step(self): calls self.forward() by default, thus no override required here