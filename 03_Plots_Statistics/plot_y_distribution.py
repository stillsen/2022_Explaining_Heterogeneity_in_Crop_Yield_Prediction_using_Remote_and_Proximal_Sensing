# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import time
import os
from copy import deepcopy

# Libs
from pytorch_lightning import seed_everything
import numpy as np
import  pandas as pd

import torch
from torch import nn

import seaborn as sns
from matplotlib import pyplot as plt
# Own modules
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
    seed_everything(42)

    ## HYPERPARAMETERS
    num_epochs = 200
    num_epochs_finetuning = 10
    lr = 0.001  # (Krizhevsky et al.2012)
    lr_finetuning = 0.0001
    momentum = 0.9  # (Krizhevsky et al.2012)
    wd = 0.0005  # (Krizhevsky et al.2012)
    classes = 1
    # batch_size = 16
    batch_size = 128  # (Krizhevsky et al.2012)
    num_folds = 4  # 9 # ranom-CV -> 1
    min_delta = 0.001
    patience = 10
    min_epochs = 200

    patch_no = 65#test_patch_no = 65
    # test_patch_no = 95
    stride = 30  # 20 is too small
    # architecture = 'baselinemodel'
    architecture = 'densenet'
    # architecture = 'short_densenet'
    # architecture = 'resnet50'
    augmentation = True
    tune_fc_only = True
    pretrained = True
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None
    scv = True
    # input_type='_grn'
    input_type = ''

    fake_labels = False
    # training_response_normalization = True
    training_response_normalization = False

    model_name = output_dirs[patch_no].split('/')[-1]
    test_patch_name = data_dirs[patch_no].split('/')[-1]
    # test_patch_name = data_dirs[patch_no].split('/')[-1] + '_grn'

    this_output_dir = output_dirs[patch_no]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_output_dir = output_dirs[patch_no]

    sets = ['train', 'val', 'test']

    predictions = dict()
    y_in = dict()
    fig_yhat, axs_yhat = plt.subplots(1, len(sets))
    fig_y, axs_y = plt.subplots(1, len(sets))
    for i, s in enumerate(sets):
        # loop over folds, last fold is for testing only
        predictions[s] = []
        y_in[s] = []
        for k in range(num_folds):
            print(f"STARTING FOLD {k}")
            # Detect if we have a GPU available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('working on device %s' % device)

            y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat_' + s + '_' + str(k) + '.pt'))
            y = torch.load(os.path.join(this_output_dir, test_patch_name+'y_' + s + '_' + str(k) + '.pt'))

            predictions[s].append(y_hat)
            y_in[s].append(y)

        all_labels_val = []
        all_k = []
        for k in range(num_folds):
            all_labels_val.extend(predictions[s][k])
            all_k.extend(np.ones(len(predictions[s][k]))*k)
        df_dict = {'l': all_labels_val, 'k': all_k}
        df = pd.DataFrame(df_dict)

        sns.violinplot(x="k", y="l", data=df, ax=axs_yhat[i])
        axs_yhat[i].set_xlabel(s)
        # fig.savefig(os.path.join(this_output_dir, s+'_folds_dist_yhat.png'))

        all_labels_val = []
        all_k = []
        for k in range(num_folds):
            all_labels_val.extend(y_in[s][k])
            all_k.extend(np.ones(len(y_in[s][k]))*k)
        df_dict = {'l': all_labels_val, 'k': all_k}
        df = pd.DataFrame(df_dict)

        sns.violinplot(x="k", y="l", data=df, ax=axs_y[i])
        axs_y[i].set_xlabel(s)
        # fig.savefig(os.path.join(this_output_dir, s + '_folds_dist_y.png'))
    fig_yhat.savefig(os.path.join(this_output_dir, 'dist_yhat.png'))
    fig_y.savefig(os.path.join(this_output_dir, 'dist_y.png'))