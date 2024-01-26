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
from ray.tune.search.bohb import TuneBOHB


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer
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

    # def setup(self, config, datamodule=None, momentum=None, patch_no=None, architecture=None, tune_fc_only=None, pretrained=None, criterion=None, device=None, training_response_standardizer=None, workers:int=1):
    def setup(self,
              config,
              workers=None,
              datamodule=None,
              momentum=None,
              patch_no=None,
              architecture=None,
              tune_fc_only=None,
              pretrained=None,
              criterion=None,
              device=None,
              training_response_standardizer=None,
              # workers:int=1,
              state_dict=None,
              SSL=None):

        print('tuning hyper parameters on patch {}'.format(patch_no))
        # print('arch: {}'.format(architecture))
        # print('SSL: {}'.format(SSL))
        # print('state dict: {}'.format(state_dict is not None))
        # loop over folds, last fold is for testing only

        self.epoch = 0

        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device=device)
            torch.cuda.synchronize()

        datamodule.set_batch_size(batch_size=int(config['batch_size']))

        self.model_wrapper = RGBYieldRegressor_Trainer(
                                          pretrained=pretrained,
                                          tune_fc_only=tune_fc_only,
                                          architecture=architecture,
                                          criterion=criterion,
                                          device=device,
                                          workers=workers,
                                          SSL=SSL
                                               )

        # update hyperparameters
        self.model_wrapper.set_hyper_parameters(lr=config['lr'], wd=config['wd'], batch_size=config['batch_size'])
        # build and update dataloaders
        self.dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(), }
        #set dataloaders
        self.model_wrapper.set_dataloaders(dataloaders=self.dataloaders_dict)

        # print("Model SSL training: {}".format(self.model_wrapper.model.SSL_training))
        # print(self.model_wrapper.model)

        # pre-trained model incoming, i.e. domain-tuning
        if state_dict is not None:
            if architecture == 'SimSiam' or architecture == 'VICRegL':
                if SSL is None:
                    self.model_wrapper.model.SSL_training = False
                # self.model_wrapper.SSL = None
            self.model_wrapper.model.load_state_dict(state_dict)
            # reintialize fc layer's weights
            self.model_wrapper.reinitialize_fc_layers()
            # disable gradient computation for the first  layers
            self.model_wrapper.disable_all_but_fc_grads()
            # print(self.model_wrapper.model)

        # update criterion
        self.model_wrapper.set_criterion(criterion=criterion)
        # update optimizer to new hyper parameter set
        self.model_wrapper.set_optimizer()
        # data parallelize and send model to device
        self.model_wrapper.parallize_and_to_device()

    def step(self):
        # train_loss = self.model_wrapper.tune_step()
        train_loss = self.model_wrapper.train_step(epoch=self.epoch)
        self.epoch = self.epoch+1
        val_loss = self.model_wrapper.test(phase='val')
        return {"train_loss": train_loss, "val_loss": val_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        torch.save(self.model_wrapper.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint):
        self.model_wrapper.model.load_state_dict(torch.load(checkpoint))
