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
from osgeo import gdal


from PatchCROPDataModule import PatchCROPDataModule
# import importlib.util
# spec = importlib.util.spec_from_file_location("PatchCROPDataModule", "/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Source/02_DL/PatchCROPDataModule.py")
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)


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
    num_samples_per_fold = None

    this_output_dir = output_dirs[patch_no]

    # datamodule = foo.PatchCROPDataModule(input_files=input_files_rgb[patch_no], patch_id=patch_no, data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation, batch_size=1)
    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no], patch_id=patch_no, data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation, batch_size=1)
    datamodule.prepare_data(num_samples=num_samples_per_fold)

    fold_starts_x = [0, 774, 1548]
    fold_length_x = 774

    fold_starts_y = [0, 774, 1548]
    fold_starts_x = [0, 774, 1548]
    kernel_size = 224
    x_size = 2325
    y_size = 2325
    summed_yield_map = np.zeros((2, y_size, x_size))

    lower_bound: int =327

    # label_matrix = torch.tensor(gdal.Open("/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Patch_ID_73/pC_col_2020_plant_PS473_SOats_smc_Krig.tif", gdal.GA_ReadOnly).ReadAsArray())
    # feature_matrix = torch.tensor(gdal.Open(
    #     "/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Patch_ID_73/Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif",
    #     gdal.GA_ReadOnly).ReadAsArray())
    label_matrix = torch.tensor(gdal.Open(
        "/beegfs/stiller/PatchCROP_all/Data/Patch_ID_73/pC_col_2020_plant_PS473_SOats_smc_Krig.tif",
        gdal.GA_ReadOnly).ReadAsArray())
    feature_matrix = torch.tensor(gdal.Open(
        "/beegfs/stiller/PatchCROP_all/Data/Patch_ID_73/Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif",
        gdal.GA_ReadOnly).ReadAsArray())
    upper_bound_x = label_matrix.shape[1] - lower_bound
    upper_bound_y = label_matrix.shape[0] - lower_bound

    yield_map = label_matrix[lower_bound:upper_bound_y, lower_bound:upper_bound_x]
    feature_map = feature_matrix[:3, lower_bound:upper_bound_y, lower_bound:upper_bound_x]

    fig, ax = plt.subplots()

    # c = ax.pcolormesh(yield_map, cmap='jet')
    c = plt.imshow(yield_map, cmap='jet')
    fig.colorbar(c, ax=ax)
    ax.hlines(y=775, xmin=0, xmax=2325, color='k')
    ax.hlines(y=1550, xmin=0, xmax=2325, color='k')
    ax.vlines(x=775, ymin=0, ymax=2325, color='k')
    ax.vlines(x=1550, ymin=0, ymax=2325, color='k')
    ax.xaxis.tick_top()

    plt.tight_layout()
    plt.savefig(os.path.join(this_output_dir, 'yield_map_inputs.png'))
    # plt.show()

    fig, ax = plt.subplots()

    # c = ax.pcolormesh(yield_map, cmap='jet')
    # c = plt.imshow(torch.moveaxis(feature_map,0,2), cmap='jet')
    c = plt.imshow(torch.movedim(feature_map, 0, 2), cmap='jet')
    fig.colorbar(c, ax=ax)
    ax.hlines(y=775, xmin=0, xmax=2325, color='k')
    ax.hlines(y=1550, xmin=0, xmax=2325, color='k')
    ax.vlines(x=775, ymin=0, ymax=2325, color='k')
    ax.vlines(x=1550, ymin=0, ymax=2325, color='k')
    ax.xaxis.tick_top()

    plt.tight_layout()
    plt.savefig(os.path.join(this_output_dir, 'feature_map_inputs.png'))