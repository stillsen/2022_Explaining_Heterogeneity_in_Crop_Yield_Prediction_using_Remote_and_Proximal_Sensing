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

import importlib.util
spec = importlib.util.spec_from_file_location("PatchCROPDataModule", "/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Source/02_DL/PatchCROPDataModule.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)


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
    num_samples_per_fold = None

    this_output_dir = output_dirs[patch_no]

    datamodule = foo.PatchCROPDataModule(input_files=input_files_rgb[patch_no], data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation, batch_size=1)
    datamodule.prepare_data(num_samples=num_samples_per_fold)

    fold_starts_x = [0, 774, 1548]
    fold_length_x = 774

    # yield_map_r1 = np.concatenate((datamodule.splits[:3]))
    # yield_map_r2 = np.concatenate((datamodule.splits[3:6]))
    # yield_map_r3 = np.concatenate((datamodule.splits[6:9]))
    # yield_map = np.concatenate((yield_map_r1[:, 1], yield_map_r2[:, 1], yield_map_r3[:, 1]), axis=1)

    fold_starts_y = [0, 774, 1548]
    fold_starts_x = [0, 774, 1548]
    kernel_size = 224
    x_size = 2325
    y_size = 2325
    summed_yield_map = np.zeros((2, y_size, x_size))

    stack = []
    i_train = 0
    i_test = 0
    for y in range(3):
        for x in range(3):

            if y==0 and x==0:
                fold_idx = 0
            elif y == 0 and x == 1:
                fold_idx = 1
            elif y == 0 and x == 2:
                fold_idx = 2
            elif y == 1 and x == 0:
                fold_idx = 3
            elif y == 1 and x == 1:
                fold_idx = 4
            elif y == 1 and x == 2:
                fold_idx = 5
            elif y == 2 and x == 0:
                fold_idx = 6
            elif y == 2 and x == 1:
                fold_idx = 7
            elif y == 2 and x == 2:
                fold_idx = 8
            print("{}th-fold, x: {}, y: {}".format(fold_idx, x, y))
            dl = datamodule.splits[fold_idx]
            sample_idx = 0
            x_in_fold = 0
            y_in_fold = 0
            for inputs, labels in dl:
                y_offset = fold_starts_y[y]+y_in_fold*10
                x_offset = fold_starts_x[x] + x_in_fold*10
                summed_yield_map[0,
                                y_offset : y_offset + kernel_size,
                                x_offset : x_offset + kernel_size] += labels.numpy()
                summed_yield_map[1,
                                y_offset: y_offset + kernel_size,
                                x_offset: x_offset + kernel_size] += 1
                # summed_yield_map[0, y:y + kernel_size, x:x + kernel_size] += pred  # y_hat[i]
                # summed_yield_map[1, y:y + kernel_size, x:x + kernel_size] += 1
                if x_in_fold == 54:
                    y_in_fold += 1
                    x_in_fold = 0
                else:
                    x_in_fold += 1
                sample_idx += 1
                # print("{}-th sample, x: {}, y: {}".format(sample_idx, x_in_fold, y_in_fold))

    yield_map = summed_yield_map[0, :, :] / summed_yield_map[1, :, :]
    # yield_map_r1[:, 1]


    fig, ax = plt.subplots()

    # c = ax.pcolormesh(yield_map, cmap='jet')
    c = plt.imshow(yield_map, cmap='jet')
    # set the limits of the plot to the limits of the data
    # ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    ###
    ax.hlines(y=775, xmin=0, xmax=2325, color='k')
    ax.hlines(y=1550, xmin=0, xmax=2325, color='k')
    ax.vlines(x=775, ymin=0, ymax=2325, color='k')
    ax.vlines(x=1550, ymin=0, ymax=2325, color='k')
    ####
    # plt.text(31.5,np.mean(pool)-4, 'mean', rotation='vertical')
    # plt.text(2200, y_size*ratio - 100, 'test', fontsize=16)
    # plt.text(2200, y_size*ratio + 100 , 'train', fontsize=16)
    # plt.xlim((0, 2304))
    # plt.ylim((0, 2304))
    # plt.xticks([0, 2325], [origin[0], end[0]])
    # # locs, labels = plt.yticks()
    # plt.yticks([0, 2304], [origin[1], end[1]])
    ax.xaxis.tick_top()

    plt.tight_layout()
    # # plt.xticks([origin[0], end[0]])
    # # plt.yticks([origin[1], end[1]])
    plt.savefig(os.path.join(this_output_dir, 'yield_map_folds.png'))
    # plt.show()


    # sets = ['train', 'val']
    #
    # predictions = dict()
    # y_in = dict()
    # for s in sets:
    #     # loop over folds, last fold is for testing only
    #     predictions[s] = []
    #     y_in[s] = []
    #     for k in range(num_folds):
    #         print(f"STARTING FOLD {k}")
    #         # Detect if we have a GPU available
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         print('working on device %s' % device)
    #
    #         y_hat = torch.load(os.path.join(this_output_dir, 'y_hat_' + s + '_' + str(k) + '.pt'))
    #         y = torch.load(os.path.join(this_output_dir, 'y_' + s + '_' + str(k) + '.pt'))
    #
    #         predictions[s].append(y_hat)
    #         y_in[s].append(y)
    #
    #     all_labels_val = []
    #     all_k = []
    #     for k in range(num_folds):
    #         all_labels_val.extend(predictions[s][k])
    #         all_k.extend(np.ones(len(predictions[s][k]))*k)
    #     df_dict = {'l': all_labels_val, 'k': all_k}
    #     df = pd.DataFrame(df_dict)
    #
    #     fig = plt.figure()
    #     ax = sns.violinplot(x="k", y="l", data=df)
    #     fig.savefig(os.path.join(this_output_dir, s+'_folds_dist_preds.png'))
    #     # fig.savefig(os.path.join(output_root, s + '_folds_dist_yforpreds.png'))
    #
    #     all_labels_val = []
    #     all_k = []
    #     for k in range(num_folds):
    #         all_labels_val.extend(y_in[s][k])
    #         all_k.extend(np.ones(len(y_in[s][k]))*k)
    #     df_dict = {'l': all_labels_val, 'k': all_k}
    #     df = pd.DataFrame(df_dict)
    #
    #     fig = plt.figure()
    #     ax = sns.violinplot(x="k", y="l", data=df)
    #     # fig.savefig(os.path.join(output_root, s+'_folds_dist_preds.png'))
    #     fig.savefig(os.path.join(this_output_dir, s + '_folds_dist_yforpreds.png'))
