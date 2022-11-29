# -*- coding: utf-8 -*-
"""
script to train the baseline model following (Nevavuori et al.2019) and (Krizhevsky et al.2012)
"""

# Built-in/Generic Imports
import os
import random

# Libs
from ctypes import c_int

import numpy as np
import pandas as pd

import time
from copy import deepcopy

import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor import RGBYieldRegressor
# from MC_YieldRegressor import MCYieldRegressor
from directory_listing import output_dirs, data_dirs, input_files_rgb


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

class TuneYieldRegressor(tune.Trainable):

    def setup(self, config, datamodule=None, momentum=None, patch_no=None, architecture=None, tune_fc_only=None, pretrained=None, criterion=None):

        print('tuning hyper parameters on patch {}'.format(patch_no))
        # loop over folds, last fold is for testing only

        # Detect if we have a GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda")
        print('working on device %s' % self.device)

        datamodule.set_batch_size(batch_size=int(config['batch_size']))
        dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(), 'test': datamodule.test_dataloader()}
        #### TUNE LAST LAYER, FREEZE BEFORE::

        self.model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                          device=self.device,
                                          # lr=lr,
                                          lr=config['lr'],
                                          momentum=momentum, #config['momentum'],
                                          wd=config['wd'],
                                          # wd=wd,
                                          # k=num_folds,
                                          pretrained=pretrained,
                                          tune_fc_only=tune_fc_only,
                                          model=architecture,
                                          training_response_standardizer=datamodule.training_response_standardizer,
                                          criterion=criterion,
                                               )

        # Send the model to GPU
        if torch.cuda.device_count() > 1:
            self.model_wrapper.model = nn.DataParallel(self.model_wrapper.model)
        self.model_wrapper.model.to(self.device)

    def step(self):
        train_loss = self.model_wrapper.train()
        val_loss = self.model_wrapper.test()
        return {"train_loss": train_loss, "val_loss": val_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        torch.save(self.model_wrapper.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        self.model_wrapper.model.load_state_dict(torch.load(checkpoint_path))
