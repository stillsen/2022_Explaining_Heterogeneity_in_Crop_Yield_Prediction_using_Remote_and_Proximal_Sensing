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

from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import pickle

# Own modules
from PatchCROPDataModule import PatchCROPDataModule, KFoldLoop
from RGBYieldRegressor import RGBYieldRegressor
from MCYieldRegressor import Resnet_multichannel



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
    # Test for flower strip patch
    # output_dir = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output'
    output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_12'
    # data_dirs = {12: '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Patch_ID_12'}
    data_dirs = {12: '/beegfs/stiller/PatchCROP_all/Data/Patch_ID_12'}

    input_files = {12: {
        'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_12.tif',
                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_12.tif',
                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_12.tif',
                                                     'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}}
    input_files_rgb = {12: {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}}

    # Test for non flower strip patch
    data_dirs[39] = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Patch_ID_39'
    input_files[39] = {
        'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_39.tif',
                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}
    input_files_rgb[39] = {
        'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}
    # # Test for Lupine
    data_dirs['Lupine'] = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Lupine'
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


    model = RGBYieldRegressor()
    # model = Resnet_multichannel(num_in_channels=6, encoder_depth=50)
    datamodule = PatchCROPDataModule(input_files=input_files_rgb[12], data_dir=data_dirs[12], stride=10, workers=os.cpu_count())
    callbacks = [PrintCallback(),
                 # EarlyStopping(monitor="val_loss", min_delta=0.0, patience=3)
                 ]
    # default logger used by trainer
    logger = TensorBoardLogger(save_dir=output_dir, version=1, name="lightning_logs")

    trainer = Trainer(
        max_epochs=40, # general
        num_sanity_val_steps=0,
        devices=1,
        accelerator="auto",
        # accelerator="GPU",
        # auto_lr_find=True,
        callbacks=callbacks,
        default_root_dir=output_dir,
        weights_save_path=output_dir,
        logger=logger,

        # fast_dev_run=True, # debugging
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2,
        # overfit_batches=1,

        num_processes=1, # HPC
        # prepare_data_per_node=False,
        # strategy="ddp",
    )
    internal_fit_loop = trainer.fit_loop
    # actually nine folds, but last is always test set
    trainer.fit_loop = KFoldLoop(num_folds=num_folds, export_path=output_dir)
    trainer.fit_loop.connect(internal_fit_loop)
    # find best lr
    # trainer.tune(model, datamodule=datamodule)
    # trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
    trainer.fit(model, datamodule=datamodule)

    pickle.dump(datamodule.test_dataset, open(os.path.join(output_dir,'testset.pickle'), 'wb'))

    ##### later
    # trainer.test(model, dataloaders=datamodule.test_dataloader())

    #
    #
    # local_r2_scores = [r2_score(trainer.predictions[idx,:].detach().numpy(), trainer.y.detach().numpy()) for idx in range(num_folds)]
    # print(local_r2_scores)
    #
    # global_predictions = trainer.predictions.flatten().detach().numpy()
    # global_y = np.repeat(trainer.y.detach().numpy(), num_folds)
    # global_r2 = r2_score(global_y, global_predictions)
    # print(global_r2)