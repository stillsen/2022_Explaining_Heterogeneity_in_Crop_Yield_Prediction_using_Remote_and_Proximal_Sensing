# -*- coding: utf-8 -*-
"""
make predictions on internal sets, i.e. train and validation set using trained model
"""

# Built-in/Generic Imports
import os, random

# Libs
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn
from collections import OrderedDict

# from pytorch_lightning import seed_everything#, Trainer
# from pytorch_lightning.callbacks import Callback, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# Own modules
from PatchCROPDataModule import PatchCROPDataModule#, KFoldLoop#, SpatialCVModel
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


if __name__ == "__main__":
    # seed_everything(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ## HYPERPARAMETERS
    num_epochs = 200
    num_epochs_finetuning = 10
    lr = 0.001 # (Krizhevsky et al.2012)
    lr_finetuning = 0.0001
    momentum = 0.9 # (Krizhevsky et al.2012)
    wd = 0.0005 # (Krizhevsky et al.2012)
    classes = 1
    # batch_size = 16
    batch_size = 1
    num_folds = 4
    min_delta = 0.001
    patience = 10
    min_epochs = 200
    repeat_trainset_ntimes = 1

    # patch_no = 73
    # patch_no = 68
    patch_no = 76
    # patch_no = 65
    stride = 30 # 20 is too small
    architecture = 'baselinemodel'
    # architecture = 'densenet'
    # architecture = 'short_densenet'
    # architecture = 'resnet18'
    augmentation = False
    tune_fc_only = True
    pretrained = False
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None
    validation_strategy = 'SCV'
    # scv = False

    fake_labels = False
    # training_response_normalization = True
    training_response_normalization = False

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('working on device %s' % device)
    if device == 'cpu':
        workers = os.cpu_count()
    else:
        workers = 1#torch.cuda.device_count()
        print('\twith {} workers'.format(workers))


    this_output_dir = output_dirs[patch_no] + '_baselinemodel_SCV_no_test_L1_ALB_TR1_E2000'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                     patch_id=patch_no,
                                     data_dir=data_dirs[patch_no],
                                     stride=stride,
                                     workers=workers,
                                     augmented=augmentation,
                                     input_features=features,
                                     batch_size=batch_size,
                                     validation_strategy=validation_strategy,
                                     fake_labels=fake_labels,
                                     )
    checkpoint_paths = ['model_f0.ckpt',
                        'model_f1.ckpt',
                        'model_f2.ckpt',
                        'model_f3.ckpt',
                        ]
    if 'SSL' in this_output_dir:
        checkpoint_paths = ['model_f0_domain-tuning.ckpt',
                            'model_f1_domain-tuning.ckpt',
                            'model_f2_domain-tuning.ckpt',
                            'model_f3_domain-tuning.ckpt',
                            ]
    checkpoint_paths = [this_output_dir+'/'+cp for cp in checkpoint_paths]

    if num_samples_per_fold == None:
        datamodule.prepare_data(num_samples=num_samples_per_fold)
    else:
        datamodule.load_subsamples(num_samples=num_samples_per_fold)

    sets = ['train', 'val',
            # 'test',
            ]
    # sets = ['val']
    global_pred = dict()
    global_y = dict()
    local_r2 = dict()
    local_r = dict()
    global_r2 = dict()
    global_r = dict()

    for s in sets:
        global_pred[s] = []
        global_y[s] = []
        local_r2[s] = []
        local_r[s] = []

        for k in range(num_folds):
            print('predictions for fold {}'.format(k))
            # load data
            datamodule.setup_fold(fold=k, training_response_standardization=training_response_normalization)
            dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(),
                                # 'test': datamodule.test_dataloader(),
                                }

            # set up model wrapper and input data
            model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                              device=device,
                                              lr=lr,
                                              momentum=momentum,
                                              wd=wd,
                                              # k=num_folds,
                                              pretrained=pretrained,
                                              tune_fc_only=tune_fc_only,
                                              model=architecture,
                                              training_response_standardizer=datamodule.training_response_standardizer
                                              )

            # load trained model weights with respect to possible parallelization and device
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_paths[k])
            else:
                state_dict = torch.load(checkpoint_paths[k], map_location=torch.device('cpu'))
            if workers > 1:
                model_wrapper.model = nn.DataParallel(model_wrapper.model)
            #
            # state_dict_revised = OrderedDict()
            # for key, value in state_dict.items():
            #     revised_key = key.replace("module.", "")
            #     state_dict_revised[revised_key] = value

            state_dict_revised = state_dict
            model_wrapper.model.load_state_dict(state_dict_revised)

            # make prediction
            # for each fold store labels and predictions
            local_preds, local_labels = model_wrapper.predict(phase=s)

            # save labels and predictions for each fold
            torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_' + s + '_' + str(k) + '.pt'))
            torch.save(local_labels, os.path.join(this_output_dir, 'y_' + s + '_' + str(k) + '.pt'))

            # for debugging, save labels and predictions in df
            y_yhat_df = pd.DataFrame({'y': local_labels, 'y_hat': local_preds})
            y_yhat_df.to_csv(os.path.join(this_output_dir, 'y-y_hat_{}_{}.csv'.format(s, k)), encoding='utf-8')
