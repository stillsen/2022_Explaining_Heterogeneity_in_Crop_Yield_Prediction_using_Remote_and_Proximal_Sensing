# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import os, random, time, gc


# Libs
from ctypes import c_int

import numpy as np
import pandas as pd

import time
from copy import deepcopy

# import ray
# from ray import tune
# from ray.tune.schedulers import HyperBandForBOHB
# from ray.tune.search.bohb import TuneBOHB


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
# from RGBYieldRegressor_Trainer import RGBYieldRegressor_Trainer
# from TuneYieldRegressor import TuneYieldRegressor
from ModelSelection_and_Validation import ModelSelection_and_Validation

from directory_listing import output_dirs, data_dirs, input_files_rgb

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


# ray.init(local_mode=True)
seed = 42
# seed_everything(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

## HYPERPARAMETERS
num_epochs = 1000
# lr_1 = None
# lr_2 = None
lr_finetuning = None
# lr_finetuning = 0.0005
momentum = 0.9 # (Krizhevsky et al.2012)
# wd = 0.0005 # (Krizhevsky et al.2012)
# wd = 0.003593991109916679
# wd_1 = None
# wd_2 = None
classes = 1
# batch_size_1 = None
# batch_size_2 = None
# batch_size = 256 # tuning 1
num_folds = 4#9 # ranom-CV -> 1
min_delta = 0.01 # aka 1%
patience = 10
min_epochs = 1000
# repeat_trainset_ntimes_1 = 1
# repeat_trainset_ntimes_2 = 10

# patch_no = 73
patch_no = 65
stride = 30 # 20 is too small
# architecture = 'baselinemodel'
# architecture = 'densenet'
architecture = 'resnet18'
# augmentation_1 = False
# augmentation_2 = True
# tune_fc_only_1 = False
# tune_fc_only_2 = True
pretrained = False
features = 'RGB'
# features = 'RGB+'
num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
validation_strategy = 'SCV_no_test' # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
# scv = False
# fake_labels_1 = True
# fake_labels_2 = False
# training_response_normalization = True
training_response_normalization = False


this_output_dir = output_dirs[patch_no]+'_'+architecture+'_'+validation_strategy+'_SSL_ALB'+'_E'+str(num_epochs)+'_resetW'

# check if exists, -> error,
# else create
if not os.path.exists(this_output_dir):
    print('creating: \t {}'.format(this_output_dir))
    os.mkdir(this_output_dir)
else:
    warnings.warn("{} directory exists. WILL OVERRIDE.".format(this_output_dir))

print('working on patch {}'.format(patch_no))
# loop over folds, last fold is for testing only

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('working on device %s' % device)
if device == 'cpu':
    workers = os.cpu_count()
else:
    workers = torch.cuda.device_count()
    print('\twith {} workers'.format(workers))

print('Setting up data in {}'.format(data_dirs[patch_no]))

# dictionary for training strategy::
# 1) self-supervised pretraining
# 2) domain-tuning
training_strategy = ['self-supervised','domain-tuning']
training_strategy_params = {
    training_strategy[0]: {
            'tune_fc_only': False,
            'fake_labels': True,
            'augmentation' : True,
            'lr' : None,
            'wd' : None,
            'batch_size' : None,
    },
    training_strategy[1]: {
        'tune_fc_only': True,
        'fake_labels': False,
        'augmentation' : True,
        'lr' : None,
        'wd' : None,
        'batch_size' : None,
    }
}
tune_name = dict()
tune_name['self-supervised'] = ''
tune_name['domain-tuning'] = ''

criterion = {training_strategy[0]: nn.MSELoss(reduction='mean'),
             training_strategy[1] : nn.L1Loss(reduction='mean')}

datamodule = dict()
datamodule['self-supervised'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                    patch_id=patch_no,
                                                    data_dir=data_dirs[patch_no],
                                                    stride=stride,
                                                    workers=workers,
                                                    augmented=training_strategy_params['self-supervised']['augmentation'],
                                                    input_features=features,
                                                    batch_size=training_strategy_params['self-supervised']['batch_size'],
                                                    validation_strategy=validation_strategy,
                                                    fake_labels=training_strategy_params['self-supervised']['fake_labels'],
                                                    )
datamodule['self-supervised'].prepare_data(num_samples=num_samples_per_fold)
datamodule['domain-tuning'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                  patch_id=patch_no,
                                                  data_dir=data_dirs[patch_no],
                                                  stride=stride,
                                                  workers=workers,
                                                  augmented=training_strategy_params['domain-tuning']['augmentation'],
                                                  input_features=features,
                                                  batch_size=training_strategy_params['domain-tuning']['batch_size'],
                                                  validation_strategy=validation_strategy,
                                                  fake_labels=training_strategy_params['domain-tuning']['fake_labels'],
                                                  )
datamodule['domain-tuning'].prepare_data(num_samples=num_samples_per_fold)

cv_models = ModelSelection_and_Validation(num_folds=num_folds,
                                          pretrained=pretrained,
                                          this_output_dir=this_output_dir,
                                          datamodule=datamodule,
                                          training_response_normalization=training_response_normalization,
                                          validation_strategy=validation_strategy,
                                          patch_no=patch_no,
                                          seed=seed,
                                          num_epochs=num_epochs,
                                          patience=patience,
                                          min_delta=min_delta,
                                          min_epochs=min_epochs,
                                          momentum=momentum,
                                          architecture=architecture,
                                          only_tune_hyperparameters=False,
                                          device=device,
                                          workers=workers
                                          )
cv_models.train_and_tune_SSL(start=0)