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

    output_dirs = dict()
    data_dirs = dict()
    input_files = dict()
    input_files_rgb = dict()

    # data_root = '/beegfs/stiller/PatchCROP_all/Data/'
    data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
    # output_root = '/beegfs/stiller/PatchCROP_all/Output/'
    output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

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
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB+_densenet_augmented_custom')
    output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_not_augmented_custom_btf_s1000')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_not_augmented_custom_btf_s3000')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_augmented_custom_s2000')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_densenet')

    # data_dirs[73] = os.path.join(data_root, 'Patch_ID_73_0307')
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
    num_epochs = 1
    lr = 0.01
    momentum = 0.8
    wd = 0.01
    classes = 1
    batch_size = 20
    num_folds = 9

    patch_no = 73
    architecture = 'densenet'
    # architecture = 'resnet50'
    augmentation = False
    tune_fc_only = False
    features = 'RGB'
    # features = 'RGB+'
    num_samples = None

    this_output_dir = output_dirs[patch_no]

    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no], data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation)
    datamodule.prepare_data()

    # loop over folds, last fold is for testing only
    k_sets = []
    train_sets = []
    for k in range(num_folds):
        print(f"STARTING FOLD {k}")
        # Detect if we have a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('working on device %s' % device)

        if features == 'RGB':
            model_wrapper = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
        elif features == 'RGB+':
            model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)

        datamodule.setup_fold_index(k)
        dataloaders_dict = {'train': datamodule.train_dataloader(num_samples=num_samples), 'val': datamodule.val_dataloader(num_samples=num_samples)}

        # Send the model to GPU
        model_wrapper.model.to(device)

        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        # Train and evaluate
        print('training for {} epochs'.format(num_epochs))
        train_set, val_set = train_model(model_wrapper.model, dataloaders_dict, loss_fn, optimizer, delta=0.1, device=device, num_epochs=num_epochs, )

        k_sets.append(val_set)
        train_sets.append(train_set)

        print('step')

    all_labels_val = []
    all_k = []
    for k in range(9):
        all_labels_val.extend(k_sets[k])
        all_k.extend(np.ones(len(k_sets[0]))*k)
    df_dict = {'l': all_labels_val, 'k': all_k}
    df = pd.DataFrame(df_dict)

    fig = plt.figure()
    ax = sns.violinplot(x="k", y="l", data=df)
    fig.savefig(os.path.join(output_root, 'val_folds_dist_in.png'))

    all_labels_train = []
    all_k = []

    for k in range(9):
        all_labels_train.extend(train_sets[k])
        all_k.extend(np.ones(len(train_sets[k])) * k)
    df_dict = {'l': all_labels_train, 'k': all_k}
    df = pd.DataFrame(df_dict)

    fig = plt.figure()
    ax = sns.violinplot(x="k", y="l", data=df)
    fig.savefig(os.path.join(output_root, 'train_folds_dist_in.png'))
