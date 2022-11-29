# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports

# Libs
from pytorch_lightning import seed_everything, Trainer
from sklearn.metrics import r2_score
import numpy as np
import os
from copy import deepcopy

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pickle

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

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")



#############################################################################################
#                           Step 5 / 5: Connect the KFoldLoop to the Trainer                #
# After creating the `KFoldDataModule` and our model, the `KFoldLoop` is being connected to #
# the Trainer.                                                                              #
# Finally, use `trainer.fit` to start the cross validation training.                        #
#############################################################################################


if __name__ == "__main__":
    seed_everything(42)

    output_dirs = dict()
    data_dirs = dict()
    input_files = dict()
    input_files_rgb = dict()

    data_root = '/beegfs/stiller/PatchCROP_all/Data/'
    # data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
    output_root = '/beegfs/stiller/PatchCROP_all/Output/'
    # output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

    ## Patch 12
    output_dirs[12] = os.path.join(output_root,'Patch_ID_12')
    data_dirs[12] = os.path.join(data_root,'Patch_ID_12')
    input_files[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_12.tif',
                                                                    'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']
                       }
    input_files_rgb[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}


    ## Patch 73
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_densenet_0307')
    output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB+_densenet_augmented')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_augmented')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_densenet')

    # data_dirs[73] = os.path.join(data_root, 'Patch_ID_73_0307')
    # data_dirs[73] = os.path.join(data_root, 'Patch_ID_73')
    data_dirs[73] = os.path.join(data_root, 'Patch_ID_73_NDVI')

    input_files[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
                       }
    # input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']}
    input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': [#'Tempelberg_soda3D_03072020_transparent_mosaic_group1_Patch_ID_73.tif',
                                                                          'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif',
                                                                          'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
                                                                          ]}

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

    ## Patch 119
    output_dirs[119] = os.path.join(output_root, 'Patch_ID_119_RGB_densenet_augmented')
    data_dirs[119] = os.path.join(data_root, 'Patch_ID_119')

    input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif']}
    # input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif',
    #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_119.tif',
    #                                                                       ]}

    ## Patch 50
    output_dirs[50] = os.path.join(output_root, 'Patch_ID_50_RGB_densenet_augmented')
    data_dirs[50] = os.path.join(data_root, 'Patch_ID_50')

    input_files_rgb[50] = {'pC_col_2020_plant_PS450_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_50.tif']}
    # input_files_rgb[50] = {'pC_col_2020_plant_PS450_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_50.tif',
    #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_50.tif',
    #                                                                       ]}

    ## Patch 105
    output_dirs[105] = os.path.join(output_root, 'Patch_ID_105_RGB_densenet_augmented')
    data_dirs[105] = os.path.join(data_root, 'Patch_ID_105')

    input_files_rgb[105] = {'pC_col_2020_plant_PS4105_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_105.tif']}
    # input_files_rgb[105] = {'pC_col_2020_plant_PS4105_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_105.tif',
    #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_105.tif',
    #                                                                       ]}

    ## Patch 19
    output_dirs[19] = os.path.join(output_root, 'Patch_ID_19_RGB_densenet_augmented')
    data_dirs[19] = os.path.join(data_root, 'Patch_ID_19')

    input_files_rgb[19] = {'pC_col_2020_plant_PS419_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_19.tif']}
    # input_files_rgb[19] = {'pC_col_2020_plant_PS419_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_19.tif',
    #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_19.tif',
    #                                                                       ]}

    ## Patch 95
    output_dirs[95] = os.path.join(output_root, 'Patch_ID_95_RGB_densenet_augmented')
    data_dirs[95] = os.path.join(data_root, 'Patch_ID_95')

    input_files_rgb[95] = {'pC_col_2020_plant_PS495_Sun_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_95.tif']}
    # input_files_rgb[95] = {'pC_col_2020_plant_PS495_Sun_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_95.tif',
    #                                                                       'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_95.tif',
    #                                                                       ]}

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

    num_folds = 8
    this_output_dir = output_dirs[95]

    datamodule = PatchCROPDataModule(input_files=input_files_rgb[95], data_dir=data_dirs[95], stride=10, workers=os.cpu_count(), augmented=True)
    datamodule.prepare_data()

    # loop over folds, last fold is for testing only
    for k in range(num_folds):
        print(f"STARTING FOLD {k}")
        lightningmodule = RGBYieldRegressor(pretrained=True, tune_fc_only=True, model='densenet')
        # lightningmodule = MCYieldRegressor(pretrained=True, tune_fc_only=False, model='densenet')
        logger = TensorBoardLogger(save_dir=this_output_dir, version=k, name="lightning_logs")
        callbacks = [PrintCallback(),
                     EarlyStopping(monitor="val_loss",
                                   min_delta=.0,
                                   check_on_train_epoch_end=True,
                                   patience=10,
                                   check_finite=True,
                                   # stopping_threshold=1e-4,
                                   mode='min'),
                     ModelCheckpoint(dirpath=this_output_dir,
                                     filename='model_'+str(k)+'_{epoch}.pt',
                                     monitor='val_loss')
                     ]

        trainer = Trainer(
            max_epochs=50,  # general
            num_sanity_val_steps=0,
            devices=1,
            accelerator="auto",
            # accelerator="GPU",
            # auto_lr_find=True,
            callbacks=callbacks,
            default_root_dir=this_output_dir,
            weights_save_path=this_output_dir,
            logger=logger,

            # fast_dev_run=True,  # debugging
            # limit_train_batches=2,
            # limit_val_batches=2,
            # limit_test_batches=2,
            # overfit_batches=1,

            num_processes=1,  # HPC
            # prepare_data_per_node=False,
            # strategy="ddp",
        )
        datamodule.setup_fold_index(k)
        trainer.fit(lightningmodule, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
        # trainer automatically saves statedict of last epoch, with early stopping this is the "best model"
        # trainer.save_checkpoint(os.path.join(this_output_dir, f"model.{k}.pt")) # -->> there is an automatic save, so not needed


#####################################
# Due to bugs in PyTorch Lighnting implementation I abandon outer fit loop implementation for
#  Spatial Cross Valiation
####################################
    #
    # model = RGBYieldRegressor(pretrained=True, tune_fc_only=True)
    # datamodule = PatchCROPDataModule(input_files=input_files_rgb[73], data_dir=data_dirs[73], stride=20, workers=os.cpu_count())
    # callbacks = [PrintCallback(),
    #              EarlyStopping(monitor="val_loss",
    #                            min_delta=100.0,
    #                            check_on_train_epoch_end=True,
    #                            patience=1,
    #                            check_finite=True,
    #                            stopping_threshold=1e-4,
    #                            mode='min'),
    #              ]
    # # default logger used by trainer
    # logger = TensorBoardLogger(save_dir=this_output_dir, version=1, name="lightning_logs")
    #
    # trainer = Trainer(
    #     max_epochs=20, # general
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
    #     # fast_dev_run=True, # debugging
    #     # limit_train_batches=2,
    #     # limit_val_batches=2,
    #     # limit_test_batches=2,
    #     # overfit_batches=1,
    #
    #     num_processes=1, # HPC
    #     # prepare_data_per_node=False,
    #     # strategy="ddp",
    # )
    # internal_fit_loop = trainer.fit_loop
    # # actually nine folds, but last is always test set
    # trainer.fit_loop = KFoldLoop(num_folds=num_folds, export_path=this_output_dir)
    # trainer.fit_loop.connect(internal_fit_loop)
    # # find best lr
    # # trainer.tune(model, datamodule=datamodule)
    # # trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
    # trainer.fit(model, datamodule=datamodule)
    #
    ##### later
    # trainer.test(model, dataloaders=datamodule.test_dataloader())
###################################

