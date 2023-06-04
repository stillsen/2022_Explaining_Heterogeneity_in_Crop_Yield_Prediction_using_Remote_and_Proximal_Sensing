# -*- coding: utf-8 -*-
"""
script to train the baseline model following (Nevavuori et al.2019) and (Krizhevsky et al.2012)
"""

# Built-in/Generic Imports
import os, random, time, gc

# Libs
import numpy as np
import pandas as pd

import time
from copy import deepcopy

import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.air import session
from ray.air.checkpoint import Checkpoint


import torch
from torch import nn

import warnings

# Own modules
from PatchCROPDataModule import PatchCROPDataModule
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
# num_epochs_finetuning = 10
# lr = 0.001 # (Krizhevsky et al.2012)
# lr = 0.012234672196538655
lr = None
# lr_finetuning = 0.0005
momentum = 0.9 # (Krizhevsky et al.2012)
# wd = 0.0005 # (Krizhevsky et al.2012)
# wd = 0.003593991109916679
wd = None
classes = 1
# batch_size = 16
# batch_size = None
batch_size = None # tuning 1
num_folds = 4#9 # ranom-CV -> 1
min_delta = 0.01 # aka 1%
patience = 10
min_epochs = 1000
duplicate_trainset_ntimes = 1

# patch_no = 73
patch_no = 76
stride = 30 # 20 is too small
architecture = 'baselinemodel'
# architecture = 'densenet'
# architecture = 'resnet18'
augmentation = True
tune_fc_only = False
pretrained = False
features = 'RGB'
# features = 'RGB+'
num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
validation_strategy = 'RCV' # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
# scv = False
fake_labels = False
# training_response_normalization = True
training_response_normalization = False
# criterion = nn.L1Loss(reduction='mean')
criterion = nn.MSELoss(reduction='mean')

# this_output_dir = output_dirs[patch_no]+'_'+architecture+'_'+validation_strategy+'_L1_ALB_TR'+str(duplicate_trainset_ntimes)+'_E'+str(num_epochs)
this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_L2_cycle' + '_E' + str(num_epochs)
# this_output_dir = output_dirs[patch_no] + '_' + 'SSL' + '_' + validation_strategy + '_grn'

# check if exists, -> error,
# else create
if not os.path.exists(this_output_dir):
    print('creating: \t {}'.format(this_output_dir))
    os.mkdir(this_output_dir)
else:
    # raise FileExistsError("{} is a directory, cannot create new one".format(this_output_dir))
    warnings.warn("{} directory exists. WILL OVERRIDE.".format(this_output_dir))

print('working on patch {}'.format(patch_no))
# loop over folds, last fold is for testing only

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('working on device %s' % device)
if device == 'cpu':
    workers = os.cpu_count()
else:
    workers = 1#torch.cuda.device_count()
    print('\twith {} workers'.format(workers))

    print('Setting up data in {}'.format(data_dirs[patch_no]))
datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                 patch_id=patch_no,
                                 this_output_dir=this_output_dir,
                                 seed=seed,
                                 data_dir=data_dirs[patch_no],
                                 stride=stride,
                                 workers=workers,
                                 augmented=augmentation,
                                 input_features=features,
                                 batch_size=batch_size,
                                 validation_strategy=validation_strategy,
                                 fake_labels=fake_labels,
                                 )
datamodule.prepare_data(num_samples=num_samples_per_fold)

models = ModelSelection_and_Validation(num_folds=num_folds,
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
models.train_and_tune_OneStrategyModel(tune_fc_only=tune_fc_only,
                                       criterion=criterion,
                                       start=0)