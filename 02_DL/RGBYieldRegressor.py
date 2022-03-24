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
    def __init__(self, optimizer='adam', lr=1e-3, batch_size=16, tune_fc_only=False):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.criterion = nn.MSELoss(reduction='mean')

        # init a pretrained resnet
        self.model = models.resnet50(pretrained=True)
        num_filters = self.model.fc.in_features
        # self.backbone = models.densenet121(pretrained=True)
        # num_filters = self.backbone.classifier.in_features
        layers = list(self.model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 1
        self.model.fc = nn.Sequential(
            # self.feature_extractor,
            nn.ReLU(),
            nn.Linear(num_filters, num_target_classes))

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

    def training_step(self, batch, batch_idx): # torch.autograd?
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y, y_hat.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        # return loss_fn(torch.flatten(y_hat), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y, y_hat.squeeze())
        # perform logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y, y_hat.squeeze())
        # perform logging
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)


    #def prediction_step(self): calls self.forward() by default, thus no override required here