# -*- coding: utf-8 -*-
"""
script to train the baseline model following (Nevavuori et al.2019) and (Krizhevsky et al.2012)
"""

# Built-in/Generic Imports
import os, random, time, gc

# Libs
import numpy as np

from lightly.loss import VICRegLLoss
from lightly.transforms.vicregl_transform import VICRegLTransform


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
num_epochs = 10000
# num_epochs_finetuning = 10
# lr = 0.001 # (Krizhevsky et al.2012)
# lr = 0.012234672196538655
lr = None
# lr_finetuning = 0.0005
momentum = 0.9  # (Krizhevsky et al.2012)
# wd = 0.0005 # (Krizhevsky et al.2012)
# wd = 0.003593991109916679
wd = None
classes = 1
# batch_size = 16
# batch_size = None
batch_size = None  # tuning 1
num_folds = 4  # 9 # ranom-CV -> 1
min_delta = 0.01  # aka 1%
patience = 10
min_epochs = num_epochs
duplicate_trainset_ntimes = 1

# patch_no = 73
patch_no = 68
stride = 30
# stride = 112
# stride = 75
kernel_size = 224
# architecture = 'baselinemodel'
# architecture = 'densenet'
# architecture = 'resnet18'
architecture = 'VICRegLConvNext'
augmentation = True
tune_fc_only = False
pretrained = False
features = 'RGB'
# features = 'RGB+'
num_samples_per_fold = None  # subsamples? -> None means do not subsample but take whole fold
validation_strategy = 'SCV_no_test'  # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
fake_labels = False
# training_response_normalization = True
training_response_normalization = False
# criterion = nn.L1Loss(reduction='mean')
# criterion = nn.MSELoss(reduction='mean')
criterion = VICRegLLoss()

SSL_transforms = VICRegLTransform(
                                 # global_crop_size= kernel_size,
                                 # local_crop_size= 40,
                                 hf_prob=0.5,
                                 # vf_prob=0.5,
                                 # rr_prob=0.5,
                                 # min_scale=0.5,
                                 global_solarize_prob=0.2,
                                 cj_prob=0.2,
                                 cj_bright=0.1,
                                 cj_contrast=0.1,
                                 cj_hue=0.1,
                                 cj_sat=0.1,
                                 # n_global_views=2,
                                 n_local_views=4,
)


# this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICRegLConvNext_s75s75_ks' + str(kernel_size)
# this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICRegLConvNext_kernel_size_' + str(kernel_size)

this_output_dir = output_dirs[patch_no] + '_' + architecture + '_' + validation_strategy + '_E' + str(num_epochs) + 'lightly-VICRegLConvNext_kernel_size_' + str(kernel_size)+'_heating'

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
# device = torch.device("cpu")
print('working on device %s' % device)
if device == 'cpu':
    workers = os.cpu_count()
else:
    workers = 1  # torch.cuda.device_count()
    print('\twith {} workers'.format(workers))

print('Setting up data in {}'.format(data_dirs[patch_no]))
datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                 patch_id=patch_no,
                                 this_output_dir=this_output_dir,
                                 seed=seed,
                                 data_dir=data_dirs[patch_no],
                                 stride=stride,
                                 kernel_size=kernel_size,
                                 workers=workers,
                                 augmented=augmentation,
                                 input_features=features,
                                 batch_size=batch_size,
                                 validation_strategy=validation_strategy,
                                 fake_labels=fake_labels
                                 )
print('\t preparing data \n\tkernel_size: {}\tstride: {}'.format(kernel_size, stride))
datamodule.prepare_data(num_samples=num_samples_per_fold)
print('\t splitting data into folds')
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
                                       workers=workers,
                                       SSL_transforms=SSL_transforms,
                                       )
models.train_and_tune_lightlySSL(start=1, SSL_type = architecture, domain_training_enabled=True)