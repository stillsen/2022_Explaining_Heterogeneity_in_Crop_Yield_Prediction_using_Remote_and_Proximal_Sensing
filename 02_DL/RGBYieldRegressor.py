# -*- coding: utf-8 -*-
"""
Wrapper class for crop yield regression model employing a ResNet50 or densenet in PyTorch in a transfer learning approach using
RGB remote sensing observations
"""

# Built-in/Generic Imports

# Libs
import torch
import torchvision.models as models
import torch.nn as nn

# from pytorch_lightning.core.lightning import LightningModule

# Own modules


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

# class RGBYieldRegressor(LightningModule):
class RGBYieldRegressor:
    def __init__(self, k:int = 9, lr:float = 0.001, momentum:float = 0.8, wd:float = 0.01, batch_size:int = 16, pretrained:bool = True, tune_fc_only:bool = True, model: str = 'densenet'):
        # super(RGBYieldRegressor, self).__init__()

        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.batch_size = batch_size
        self.k = k

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

        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                            lr=lr,
                                            momentum=momentum,
                                            weight_decay=wd)

        self.criterion = nn.MSELoss(reduction='mean')


    def enable_grads(self):
        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = True

    def update_optimizer(self, lr, momentum, wd):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=wd)
    def update_criterion(self, loss: torch.nn.modules.loss):
        self.criterion = loss

    # def forward(self, x):
    #     return torch.flatten(self.model(x))
    #
    # def training_step(self, batch, batch_idx): # torch.autograd?
    #     x, y = batch
    #     y_hat = torch.flatten(self.model(x))
    #     loss = self.criterion(y_hat, y)
    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = torch.flatten(self.model(x))
    #     loss = self.criterion(y_hat, y)
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    #
    #
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = torch.flatten(self.model(x))
    #     loss = self.criterion(y_hat, y)
    #     self.log("test_loss", loss, prog_bar=True, logger=True)
    #
    # def predicts_step(self, batch, batch_idx, dataloader_idx=0):
    #     return self.model(batch).squeeze()
    #
    # def configure_optimizers(self):
    #     return self.optimizer(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
    #
    #
    # #def prediction_step(self): calls self.forward() by default, thus no override required here