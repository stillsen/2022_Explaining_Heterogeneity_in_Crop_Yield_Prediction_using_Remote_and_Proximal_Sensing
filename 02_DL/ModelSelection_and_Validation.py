# -*- coding: utf-8 -*-
"""
Wrapper class model selection and validation
"""

# Built-in/Generic Imports

# Libs
import time, gc, os
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torchvision.models as models
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import ResNet

import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
# Own modules
from BaselineModel import BaselineModel
from TuneYieldRegressor import TuneYieldRegressor

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

class RGBYieldRegressor_Trainer:
    def __init__(self, device, pretrained:bool = True, tune_fc_only:bool = True, architecture: str = 'resnet18', criterion=None):