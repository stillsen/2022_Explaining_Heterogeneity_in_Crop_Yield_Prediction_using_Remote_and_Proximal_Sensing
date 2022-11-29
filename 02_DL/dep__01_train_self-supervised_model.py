# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import os
import random

# Libs
import numpy as np
import pandas as pd

import time
from copy import deepcopy

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.util import inspect_serializability


import torch
from torch import nn

import warnings
# Own modules
from PatchCROPDataModule import PatchCROPDataModule
from RGBYieldRegressor import RGBYieldRegressor
from MC_YieldRegressor import MCYieldRegressor



__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


if __name__ == "__main__":
    # seed_everything(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)


    output_dirs = dict()
    data_dirs = dict()
    input_files = dict()
    input_files_rgb = dict()

    data_root = '/beegfs/stiller/PatchCROP_all/Data/'
    # data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
    output_root = '/beegfs/stiller/PatchCROP_all/Output/'
    # output_root = '../../Output/'

    ## Patch 12
    output_dirs[12] = os.path.join(output_root,'Patch_ID_12')
    data_dirs[12] = os.path.join(data_root,'Patch_ID_12')
    input_files[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']
                       }
    input_files_rgb[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}


    ## Patch 68
    # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_baselinemodel_augmented_fakelabels_fixhyperparams')
    # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams')
    output_dirs[68] = os.path.join(output_root, 'scv_ssl_P68_600')
    # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_densenet_augmented_fakelabels_tunedhyperparams')

    data_dirs[68] = os.path.join(data_root, 'Patch_ID_68')
    # data_dirs[73] = os.path.join(data_root, 'Patch_ID_68_NDVI')

    # input_files[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
    #                                                                   'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
    #                                                                   'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
    #                                                                   'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
    #                    }
    input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif']}
    # input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif',
    #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_68.tif',
    #                                                                       ]}


    ## Patch 73
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_fixhyperparams')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet3_augmented_fakelabels_fixhyperparams')
    output_dirs[73] = os.path.join(output_root, 'scv_ssm_scv')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_augmented_fakelabels_tunedhyperparams')

    data_dirs[73] = os.path.join(data_root, 'Patch_ID_73')
    # data_dirs[73] = os.path.join(data_root, 'Patch_ID_73_NDVI')

    input_files[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
                       }
    input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']}
    # input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': [#'Tempelberg_soda3D_03072020_transparent_mosaic_group1_Patch_ID_73.tif',
    #                                                                       'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif',
    #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
    #                                                                       ]}

    ## Patch 119
    output_dirs[119] = os.path.join(output_root, 'Patch_ID_119_RGB')
    # data_dirs[119] = os.path.join(data_root, 'Patch_ID_73_0307')
    # data_dirs[119] = os.path.join(data_root, 'Patch_ID_119')
    data_dirs[119] = os.path.join(data_root, 'Patch_ID_119')

    input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif']}
    # input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif',
    #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_119.tif',
    #                                                                       ]}

    ## Patch 39
    output_dirs[39] = os.path.join(output_root, 'Patch_ID_39')
    data_dirs[39] = os.path.join(data_root, 'Patch_ID_39')

    input_files[39] = {
        'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}
    input_files_rgb[39] = {
        'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}

    # # Test for Lupine
    output_dirs[39] = os.path.join(output_root, 'Lupine')
    data_dirs['Lupine'] = os.path.join(data_root, 'Lupine')
    input_files['Lupine'] = {
        'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif'],
        'pC_col_2020_plant_PS459_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
                                                     'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_59.tif'],
        'pC_col_2020_plant_PS489_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
                                                     'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_89.tif']}

    ## HYPERPARAMETERS
    num_epochs = 600
    num_epochs_finetuning = 20
    # lr = 0.001 # (Krizhevsky et al.2012)
    lr = 0.0001522166684414705 # tuning 1
    lr_finetuning = 0.00005
    momentum = 0.9 # (Krizhevsky et al.2012)
    # wd = 0.0005 # (Krizhevsky et al.2012)
    wd = 0.0001 # tuning 1
    classes = 1
    # batch_size = 16
    batch_size = 8 #(Krizhevsky et al.2012)
    # batch_size = 256 # tuning 1
    num_folds = 1#9 # ranom-CV -> 1
    min_delta = 0.01 # aka 1%
    patience = 10
    patience_fine_tuning = 10
    min_epochs = 200

    # patch_no = 73
    patch_no = 68
    stride = 10 # 20 is too small
    architecture = 'baselinemodel'
    # architecture = 'densenet'
    # architecture = 'resnet50'
    augmentation = True
    # tune_fc_only = True
    pretrained = False
    # pretrained = True
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
    scv = True
    # scv = False
    # training_response_normalization = True
    training_response_normalization = False

    print('Setting up data in {}'.format(data_dirs[patch_no]))
    this_output_dir = output_dirs[patch_no]

    # dictionary for training strategy::
    # 1) self-supervised pretraining
    # 2) domain-tuning
    training_strategy = ['self-supervised','domain-tuning']
    training_strategy_params = {
        training_strategy[0]: {
                'tune_fc_only': False,
                'fake_labels': True,
        },
        training_strategy[1]: {
            'tune_fc_only': True,
            'fake_labels': False,
        }
    }

    # Current best trial: d9362_00021 with val_loss=25.424242816816534 and parameters={'lr': 0.0001522166684414705, 'wd': 0.0001, 'batch_size': 8}

    ## set up data module for self-supervised pre-training & domain tuning

    datamodule = dict()
    datamodule['self-supervised'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                        patch_id=patch_no,
                                                        data_dir=data_dirs[patch_no],
                                                        stride=stride,
                                                        workers=os.cpu_count(),
                                                        augmented=augmentation,
                                                        input_features=features,
                                                        batch_size=batch_size,
                                                        validation_strategy=scv,
                                                        fake_labels=training_strategy_params['self-supervised']['fake_labels'],
                                                        )
    datamodule['self-supervised'].prepare_data(num_samples=num_samples_per_fold)
    datamodule['domain-tuning'] = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                                      patch_id=patch_no,
                                                      data_dir=data_dirs[patch_no],
                                                      stride=stride,
                                                      workers=os.cpu_count(),
                                                      augmented=augmentation,
                                                      input_features=features,
                                                      batch_size=batch_size,
                                                      validation_strategy=scv,
                                                      fake_labels=training_strategy_params['domain-tuning']['fake_labels'],
                                                      )
    datamodule['domain-tuning'].prepare_data(num_samples=num_samples_per_fold)


    print('working on patch {}'.format(patch_no))

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('working on device %s' % device)

    # ############################### DEBUG
    # warnings.warn('training on 1 fold', FutureWarning)
    # validation_set_pos= [[0, 0], [1162, 1162]]
    # validation_set_extend=[[1161, 1161], [1161, 1161]]
    validation_set_pos = [[1162, 0]]
    validation_set_extend = [[1161, 2323]]
    for k in range(1):
        print('#'*20)
        print(f"STARTING FOLD {k}")
        # save training statistics
        train_loss = dict()
        val_loss = dict()
        best_epoch = dict()
        # 1) self-supervised pretraining, 2) domain-tuning
        for strategy in training_strategy:
            print('Training in {}-strategy'.format(strategy))
            ## SELF-SUPERVISED PRE-TRAINING
            # sample data in folds & augment and standardize
            datamodule[strategy].setup_fold(validation_set_pos=validation_set_pos[k],
                                  validation_set_extend=validation_set_extend[k],
                                  data_set_row_extend=2323,
                                  data_set_column_extend=2323,
                                  buffer = 1,#223,
                                  training_response_standardization=training_response_normalization
                                  )
            dataloaders_dict = {'train': datamodule[strategy].train_dataloader(), 'val': datamodule[strategy].val_dataloader()}

            if strategy == 'self-supervised': # first run?
                # init model wrapper
                if features == 'RGB':
                    model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                                      device=device,
                                                      lr=lr,
                                                      momentum=momentum,
                                                      wd=wd,
                                                      # k=num_folds,
                                                      pretrained=pretrained,
                                                      tune_fc_only=training_strategy_params['self-supervised']['tune_fc_only'],
                                                      model=architecture,
                                                      training_response_standardizer=datamodule[strategy].training_response_standardizer
                                                      )
                elif features == 'RGB+':
                    model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=True, model=architecture, lr=lr, wd=wd)
                # send model to device
                # Parallelize model
                # if torch.cuda.device_count() > 1:
                #     model_wrapper.model = nn.DataParallel(model_wrapper.model)
                model_wrapper.model.to(device)
                # save init weights for fc
                model_wrapper.save_SSL_fc_weights()

            else: # domain-tuning
                # update dataloaders
                model_wrapper.set_dataloaders(dataloaders=dataloaders_dict)
                # keep model weights -> nothing to do
                # but reinitialize optimizer
                model_wrapper.set_optimizer(lr=lr_finetuning,
                                            momentum=momentum,
                                            wd=wd,
                                            )
                # reset fc weights
                model_wrapper.reset_SSL_fc_weights()
                # disable all but fc grads
                model_wrapper.disable_all_but_fc_grads()

            # train model according to a strategy
            print('training for {} epochs'.format(num_epochs))
            model_wrapper.train_model(patience=patience,
                                      min_delta=min_delta,
                                      num_epochs=num_epochs,
                                      min_epochs=min_epochs,
                                      )
            ###################################
            # debug step results here
            local_preds, local_labels = model_wrapper.predict(phase='train')
            # save labels and predictions for each fold
            torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_train_'+strategy+'.pt'))
            torch.save(local_labels, os.path.join(this_output_dir, 'y_train_' + strategy + '.pt'))

            # for debugging, save labels and predictions in df
            y_yhat_df = pd.DataFrame({'y': local_labels, 'y_hat': local_preds})
            y_yhat_df.to_csv(os.path.join(this_output_dir, 'y-y_hat_train_{}.csv'.format(strategy)), encoding='utf-8')

            local_preds, local_labels = model_wrapper.predict(phase='val')
            # save labels and predictions for each fold
            torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_val_'+strategy+'.pt'))
            torch.save(local_labels, os.path.join(this_output_dir, 'y_val_' + strategy + '.pt'))

            # for debugging, save labels and predictions in df
            y_yhat_df = pd.DataFrame({'y': local_labels, 'y_hat': local_preds})
            y_yhat_df.to_csv(os.path.join(this_output_dir, 'y-y_hat_val_{}.csv'.format(strategy)), encoding='utf-8')
            ###################################

            # save training statistics
            train_loss[strategy] = model_wrapper.train_mse_history
            val_loss[strategy] = model_wrapper.val_mse_history
            best_epoch[strategy] = model_wrapper.best_epoch

        ## FINE-TUNE ALL LAYERS:
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

        df = pd.DataFrame({'ss_train_loss': train_loss['self-supervised'],
                           'ss_val_loss': val_loss['self-supervised'],
                           'ss_best_epoch': best_epoch['self-supervised'],
                           'domain_train_loss': train_loss['domain-tuning'],
                           'domain_val_loss':val_loss['domain-tuning'],
                           'domain_best_epoch':best_epoch['domain-tuning'],
                           })
        df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) + '.csv'), encoding='utf-8')

