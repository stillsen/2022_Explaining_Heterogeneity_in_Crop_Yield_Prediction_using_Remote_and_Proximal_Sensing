# -*- coding: utf-8 -*-
"""
script to train model
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
from TuneYieldRegressor import TuneYieldRegressor
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
    # ray.init(local_mode=True)

    # seed_everything(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ## HYPERPARAMETERS
    num_epochs = 200
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
    batch_size = None
    # batch_size = 256 # tuning 1
    num_folds = 4#9 # ranom-CV -> 1
    min_delta = 0.01 # aka 1%
    patience = 10
    min_epochs = 5
    repeat_trainset_ntimes = 10

    # patch_no = 73
    patch_no = 65
    stride = 30 # 20 is too small
    # architecture = 'baselinemodel'
    # architecture = 'densenet'
    architecture = 'resnet18'
    augmentation = True
    tune_fc_only = True
    pretrained = True
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
    validation_strategy = 'SCV' # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; RCV => Random Cross Validation
    # scv = False
    fake_labels = False
    # training_response_normalization = True
    training_response_normalization = False
    # criterion = nn.MSELoss(reduction='mean')
    criterion = nn.L1Loss(reduction='mean')

    this_output_dir = output_dirs[patch_no]+'_'+architecture+'_'+validation_strategy+'_TL_L1_ES_ALB_TR'+str(repeat_trainset_ntimes)

    # # check if exists, -> error,
    # # else create
    # if not os.path.exists(this_output_dir):
    #     print('creating: \t {}'.format(this_output_dir))
    #     os.mkdir(this_output_dir)
    # else:
    #     raise FileExistsError("{} is a directory, cannot create new one".format(this_output_dir))

    print('Setting up data in {}'.format(data_dirs[patch_no]))
    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                     patch_id=patch_no,
                                     data_dir=data_dirs[patch_no],
                                     stride=stride,
                                     workers=os.cpu_count(),
                                     augmented=augmentation,
                                     input_features=features,
                                     batch_size=batch_size,
                                     validation_strategy=validation_strategy,
                                     fake_labels=fake_labels,
                                     )
    datamodule.prepare_data(num_samples=num_samples_per_fold)

    print('working on patch {}'.format(patch_no))
    # loop over folds, last fold is for testing only

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('working on device %s' % device)

    for k in range(num_folds):
        print('#'*60)
        print('Fold: {}'.format(k))
        # setup data according to folds
        # quadrants:
        # 0 1
        # 2 3
        # test val train   | fold
        #  0    1   {2,3}  |  0
        #  1    3   {0,2}  |  1
        #  3    2   {0,1}  |  2
        #  2    0   {1,3}  |  3
        datamodule.setup_fold(fold=k, training_response_standardization=training_response_normalization, repeat_trainset_ntimes=repeat_trainset_ntimes)
        # datamodule.create_debug_samples(n=20)

        ###### FROZEN CONV ######### tune hyper parameters #######################

        param_space = {
            "lr": tune.loguniform(1e-6, 1e-1),
            "wd": tune.uniform(0, 5 * 1e-3),
            "batch_size": tune.choice([8, 16, 32, 64, 128, 256, 512]),
        }
        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=100,
            reduction_factor=4,
            stop_last_trials=False,
        )
        algo = TuneBOHB()
        bohb_search = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=4)

        tune_name = 'Tuning_{}_{}_all_bayes_L1_ES_ALB_f{}_{}_TR{}'.format(architecture,validation_strategy,k,patch_no, repeat_trainset_ntimes)
        analysis = tune.run(tune.with_parameters(
                                TuneYieldRegressor,
                                momentum=momentum,
                                patch_no=patch_no,
                                architecture=architecture,
                                tune_fc_only=tune_fc_only,
                                pretrained=pretrained,
                                datamodule=datamodule,
                                criterion=criterion,
                                ),
                            checkpoint_freq=10,
                            max_failures=5,
                            # stop={"training_iteration" : 20},
                            config=param_space,
                            resources_per_trial={"gpu": 2},
                            metric='val_loss',
                            mode='min',
                            resume="AUTO",
                            search_alg=algo,
                            scheduler=bohb_hyperband,
                            num_samples=20,
                            stop={"training_iteration": 100},
                            name=tune_name,
                            )

        print('\tbest config: ', analysis.get_best_config(metric="val_loss", mode="min"))

        torch.save(analysis, os.path.join(this_output_dir, 'analysis_f{}.ray'.format(k)))

        lr = analysis.get_best_config(metric="val_loss", mode="min")['lr']
        wd = analysis.get_best_config(metric="val_loss", mode="min")['wd']
        batch_size = analysis.get_best_config(metric="val_loss", mode="min")['batch_size']
        # lr = 0.001
        # wd = 0.001
        # batch_size = 64

        ###### FROZEN CONV ######### train last layer #######################
        datamodule.set_batch_size(batch_size=batch_size)
        dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(), 'test': datamodule.test_dataloader()}

        model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                          device=device,
                                          lr=lr,
                                          momentum=momentum,
                                          wd=wd,
                                          # k=num_folds,
                                          pretrained=pretrained,
                                          tune_fc_only=tune_fc_only,
                                          model=architecture,
                                          training_response_standardizer=datamodule.training_response_standardizer,
                                          criterion=criterion,
                                          )

        # Send the model to GPU
        # if torch.cuda.device_count() > 1:
        #     model_wrapper.model = nn.DataParallel(model_wrapper.model)
        model_wrapper.model.to(device)

        # Train and evaluate
        print('training for {} epochs'.format(num_epochs))
        model_wrapper.train_model(patience=patience,
                                  min_delta=min_delta,
                                  num_epochs=num_epochs,
                                  min_epochs=min_epochs,
                                  )
        # save training stastistics (to not overwrite them)
        train_loss= model_wrapper.train_mse_history
        val_loss= model_wrapper.test_mse_history
        best_epoch= model_wrapper.best_epoch

        ###### UNFROZEN CONV ######### hyperparameter tuning #######################

        ###### UNFROZEN CONV ######### finetuning #######################
        # enable gradient computation for all layers
        model_wrapper.enable_grads()
        # set hyperparams for optimizer
        model_wrapper.set_optimizer( lr=lr_finetuning,
                                     momentum=momentum,
                                     wd=wd,
                                     )
        # reinitialize optimizer state
        model_wrapper.optimizer.load_state_dict(model_wrapper.optimizer_state_dict)

        print('fine-tuning all layers for {} epochs'.format(num_epochs_finetuning))
        model_wrapper.train_model(patience=patience_fine_tuning,
                                  min_delta=min_delta,
                                  num_epochs=num_epochs_finetuning,
                                  )

        # save best model
        torch.save(model_wrapper.model.state_dict(), os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt'))
        # save training statistics

        df = pd.DataFrame({'train_loss': train_loss,
                           'val_loss': val_loss,
                           'best_epoch': best_epoch,
                           # 'ft_val_loss':ft_val_losses,
                           # 'ft_train_loss':ft_train_losses,
                           # 'ft_best_epoch':ft_best_epoch,
                           })
        df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) + '.csv'), encoding='utf-8')