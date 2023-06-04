# -*- coding: utf-8 -*-
"""
Baseline model implemention inspired by (Nevavuori et al.2019) and (Krizhevsky et al.2012)
"""

# Built-in/Generic Imports

# Libs
import time

import torch
import torch.nn as nn
from torch.nn import ModuleDict

from collections import OrderedDict

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
class BaselineModel(nn.Module):
    def __init__(self, ) -> None:

        super(BaselineModel, self).__init__()

        # add first convolution to moduelist
        self.blocks = ModuleDict({
            'block_0' : nn.Sequential(OrderedDict([
                ('conv_0', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11,11), stride=(4,4),padding=(5,5),)),
                ('norm_0', nn.BatchNorm2d(num_features=64)),
                ('relu_0', nn.ReLU(inplace=True)),
                ('pool_0', nn.MaxPool2d(kernel_size=5, stride=5, padding=2)),
                ]))
        }) # out: 12x12x64
        # adding 5 intermediate layers
        for i in range(5):
            self.blocks.update({
            'block_{}'.format(i+1) : nn.Sequential(OrderedDict([
                ('conv_{}'.format(i+1), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1),padding=(2,2),)),
                ('norm_{}'.format(i+1), nn.BatchNorm2d(num_features=64)),
                ('relu_{}'.format(i+1), nn.ReLU(inplace=True)),
                ]))
            }) # out: 12x12x64
        # last conv
        self.blocks.update({
            'block_6': nn.Sequential(OrderedDict([
                ('conv_6',nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), )),
                ('norm_6', nn.BatchNorm2d(num_features=128)),
                ('relu_6', nn.ReLU(inplace=True)),
                ('pool_6', nn.MaxPool2d(kernel_size=(5,5), stride=(2,2), padding=(1,1))),
            ]))
        })  # out: 5x5x128
        self.regressor = nn.Sequential(OrderedDict([
            ('fc_7', nn.Linear(in_features=5*5*128, out_features=1024)),
            ('relu_7', nn.ReLU(inplace=True)),
            # ('fc_8', nn.Linear(in_features=1024, out_features=1024)),
            # ('relu_8', nn.ReLU(inplace=True)),
            ('fc_8', nn.Linear(in_features=1024, out_features=1))
            # ('fc_9', nn.Linear(in_features=1024, out_features=1))
            ]))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(7):
            out = self.blocks['block_{}'.format(i)](out)
        out = torch.flatten(out,1)
        out = self.regressor(out)
        return out
