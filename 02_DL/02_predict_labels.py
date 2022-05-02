# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import os

# Libs
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import torch
from torch.utils.data.dataloader import DataLoader

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# Own modules
from PatchCROPDataModule import PatchCROPDataModule#, KFoldLoop#, SpatialCVModel
from RGBYieldRegressor import RGBYieldRegressor
from MCYieldRegressor import MCYieldRegressor



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
    output_dirs[12] = os.path.join(output_root, 'Patch_ID_12')
    data_dirs[12] = os.path.join(data_root, 'Patch_ID_12')
    input_files[12] = {
        'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_12.tif',
                                                     'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_12.tif',
                                                     'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_12.tif',
                                                     'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']
        }
    input_files_rgb[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': [
        'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}

    ## Patch 73
    output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_not_augmented_custom_btf')
    # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB+_densenet_augmented_custom')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no], data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation, batch_size=1)

    checkpoint_paths = ['model_f0.ckpt',
                        'model_f1.ckpt',
                        'model_f2.ckpt',
                        'model_f3.ckpt',
                        'model_f4.ckpt',
                        'model_f5.ckpt',
                        'model_f6.ckpt',
                        'model_f7.ckpt',
                        'model_f8.ckpt',
                        ]
    checkpoint_paths = [output_dirs[patch_no]+'/'+cp for cp in checkpoint_paths]

    datamodule.prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sets = ['train', 'val']
    global_pred = dict()
    global_y = dict()
    local_r2 = dict()
    global_r2 = dict()
    if features == 'RGB':
        model_wrapper = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
    elif features == 'RGB+':
        model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)

    statistics_df = pd.DataFrame(columns=['Patch_ID', 'Features', 'Architecture', 'Set', 'Global_R2', 'Local_R2_F1', 'Local_R2_F2', 'Local_R2_F3', 'Local_R2_F4', 'Local_R2_F5', 'Local_R2_F6', 'Local_R2_F7', 'Local_R2_F8', 'Local_R2_F9'])
    for s in sets:
        global_pred[s] = []
        global_y[s] = []
        local_r2[s] = []
        for k in range(num_folds):
            # load model and set to eval mode
            model_wrapper.model.load_state_dict(torch.load(checkpoint_paths[k]))
            model_wrapper.model.to(device)
            model_wrapper.model.eval()
            # load data
            datamodule.setup_fold_index(k)
            if s == 'train':
                dl = datamodule.train_dataloader(num_samples=num_samples)
            elif s == 'val':
                dl = datamodule.val_dataloader(num_samples=num_samples)


            # for each fold store labels and predictions
            local_preds = []
            local_labels = []

            # loop over dataset
            for inputs, labels in dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # make prediction
                with torch.no_grad():
                    y_hat = torch.flatten(model_wrapper.model(inputs))
                # if s == 'train':
                #     print('{} - {}'.format(labels.detach().cpu().numpy(), y_hat.detach().cpu().numpy()))
                # while looping over the data loader save labels and predictions
                local_preds.extend(y_hat.detach().cpu().numpy())
                local_labels.extend(labels.detach().cpu().numpy())
            # save labels and predictions for each fold
            torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_' + s + '_' + str(k) + '.pt'))
            torch.save(local_labels, os.path.join(this_output_dir, 'y_' + s + '_' + str(k) + '.pt'))
            # save local r2 for a certain train or val set
            local_r2[s].append(r2_score(local_labels, local_preds))
            print(''+s+':: {}-fold local R2 {}'.format(k, local_r2[s][k]))
            # pool label and predictions for each set
            global_pred[s].extend(local_preds)
            global_y[s].extend(local_labels)
        global_r2[s] = r2_score(global_y[s], global_pred[s])
        print(''+s+':: global R2 {}'.format(global_r2))
        # save predictions to CSV
        # statistics_df = pd.read_csv(os.path.join(output_root, 'Performance_all_Patches_RGB.csv'))
        statistics_df.loc[-1] = [patch_no, features, architecture, s, global_r2[s], local_r2[s][0], local_r2[s][1], local_r2[s][2], local_r2[s][3], local_r2[s][4], local_r2[s][5], local_r2[s][6], local_r2[s][7], local_r2[s][8]]
        statistics_df.index = statistics_df.index + 1
        statistics_df.sort_index()
    statistics_df.to_csv(os.path.join(this_output_dir, 'prediction_statistics.csv'), encoding='utf-8')