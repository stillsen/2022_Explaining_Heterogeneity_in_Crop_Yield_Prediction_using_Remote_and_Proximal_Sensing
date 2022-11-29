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
    output_dirs[68] = os.path.join(output_root, 'P_68_densenet')
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
    output_dirs[73] = os.path.join(output_root, 'scv_dense_fc3_lr001')
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
    num_epochs = 100
    num_epochs_finetuning = 10
    # lr = 0.001 # (Krizhevsky et al.2012)
    lr = 0.00019205490108964047
    lr_finetuning = 0.00005
    momentum = 0.9 # (Krizhevsky et al.2012)
    # wd = 0.0005 # (Krizhevsky et al.2012)
    wd = 0.004147447769064703
    classes = 1
    batch_size = 64
    # batch_size = 256 # tuning 1
    num_folds = 1#9 # ranom-CV -> 1
    min_delta = 0.01 # aka 1%
    patience = 10
    patience_fine_tuning = 10
    min_epochs = 100

    # patch_no = 73
    patch_no = 68
    stride = 30 # 20 is too small
    # architecture = 'baselinemodel'
    architecture = 'densenet'
    # architecture = 'resnet50'
    augmentation = True
    tune_fc_only = True
    # pretrained = False
    pretrained = True
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
    validation_strategy = 'SHOV'
    # scv = False
    fake_labels = False
    # training_response_normalization = True
    training_response_normalization = False

    this_output_dir = output_dirs[patch_no]

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

    # ############################### DEBUG
    # warnings.warn('training on 1 fold', FutureWarning)
    # validation_set_pos= [[0, 0], [1162, 1162]]
    # validation_set_extend=[[1161, 1161], [1161, 1161]]
    validation_set_pos = [[1162, 0]]
    validation_set_extend = [[1161, 2323]]
    for k in range(1):
        print('#'*20)
        print(f"STARTING FOLD {k}")


        # data
        datamodule.setup_fold(validation_set_pos=validation_set_pos[k],
                              validation_set_extend=validation_set_extend[k],
                              data_set_row_extend=2323,
                              data_set_column_extend=2323,
                              buffer = 1,#223,
                              training_response_standardization=training_response_normalization
                              )
        dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader()}

        #### TRAIN LAST LAYER ONLY, FREEZE BEFORE::
        if features == 'RGB':
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
        elif features == 'RGB+':
            model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=True, model=architecture, lr=lr, wd=wd)

        # # Parallelize model
        # if torch.cuda.device_count() > 1:
        #     model_wrapper.model = nn.DataParallel(model_wrapper.model)
        model_wrapper.model.to(device)

        # for param in model_wrapper.model.parameters(): print(param.requires_grad)

        # warnings.warn('training missing', FutureWarning)
        # Train and evaluate
        print('training for {} epochs'.format(num_epochs))
        model_wrapper.train_model(patience=patience,
                                  min_delta=min_delta,
                                  num_epochs=num_epochs,
                                  min_epochs=min_epochs,
                                  )
        # best_model, val_losses, train_losses, best_epoch = train_model(model=model_wrapper.model,
        #                                                                dataloaders=dataloaders_dict,
        #                                                                criterion=model_wrapper.criterion,
        #                                                                optimizer=model_wrapper.optimizer,
        #                                                                delta=0,
        #                                                                patience=10,
        #                                                                device=device,
        #                                                                num_epochs=num_epochs,
        #                                                                )
        # save training stastistics (to not overwrite them)
        train_loss= model_wrapper.train_mse_history
        val_loss= model_wrapper.val_mse_history
        best_epoch= model_wrapper.best_epoch

        # warnings.warn('fine-tuning missing', FutureWarning)
        # warnings.warn('fine-tuning: optimizer needs state_dict loaded', FutureWarning)

        # #### Fine-tuning all layers::
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
        # best_model, ft_val_losses, ft_train_losses, ft_best_epoch = train_model(model=model_wrapper.model,
        #                                                                dataloaders=dataloaders_dict,
        #                                                                criterion=model_wrapper.criterion,
        #                                                                optimizer=model_wrapper.optimizer,
        #                                                                delta=0,
        #                                                                patience=5,
        #                                                                device=device,
        #                                                                num_epochs=num_epochs_finetuning,
        #                                                                )


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

        ## compare models
        # load_model = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
        # load_model.model.load_state_dict(torch.load(os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt')))
        # load_model.model.to(device)
        # load_model.model.eval()
        # compare_models(best_model, load_model.model)

        # ###########################################################
        #         child_counter = 0
        #         for child in list(model_wrapper.model.children())[:-1]:
        #             print(" child", child_counter, "is:")
        #             print(child)
        #             child_counter += 1
        # ###########################################################

        ## PYTORCH LIGHTNING IMPLEMENTATION - currently not working
        #
        # logger = TensorBoardLogger(save_dir=this_output_dir, version=k, name="lightning_logs")
        # callbacks = [PrintCallback(),
        #              EarlyStopping(monitor="val_loss",
        #                            min_delta=.0,
        #                            check_on_train_epoch_end=True,
        #                            patience=10,
        #                            check_finite=True,
        #                            # stopping_threshold=1e-4,
        #                            mode='min'),
        #              ModelCheckpoint(dirpath=this_output_dir,
        #                              filename='model_'+str(k)+'_{epoch}.pt',
        #                              monitor='val_loss')
        #              ]
        #
        # trainer = Trainer(
        #     max_epochs=50,  # general
        #     num_sanity_val_steps=0,
        #     devices=1,
        #     accelerator="auto",
        #     # accelerator="GPU",
        #     # auto_lr_find=True,
        #     callbacks=callbacks,
        #     default_root_dir=this_output_dir,
        #     weights_save_path=this_output_dir,
        #     logger=logger,
        #
        #     # fast_dev_run=True,  # debugging
        #     # limit_train_batches=2,
        #     # limit_val_batches=2,
        #     # limit_test_batches=2,
        #     # overfit_batches=1,
        #
        #     num_processes=1,  # HPC
        #     # prepare_data_per_node=False,
        #     # strategy="ddp",
        # )
        #
        # trainer.fit(lightningmodule, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
        # # trainer automatically saves statedict of last epoch, with early stopping this is the "best model"
        # # trainer.save_checkpoint(os.path.join(this_output_dir, f"model.{k}.pt")) # -->> there is an automatic save, so not needed
        #
        #
