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
from torch import nn
from collections import OrderedDict

from pytorch_lightning import seed_everything#, Trainer
# from pytorch_lightning.callbacks import Callback, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# Own modules
from PatchCROPDataModule import PatchCROPDataModule#, KFoldLoop#, SpatialCVModel
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


# class PrintCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         print("Training is started!")
#     def on_train_end(self, trainer, pl_module):
#         print("Training is done.")

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
    # output_root = '../../Output/'

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

    # # Patch 68
    # output_dirs[68] = os.path.join(output_root, 'P_68_densenet')
    # data_dirs[68] = os.path.join(data_root, 'Patch_ID_68')
    # input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif']}
    # input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif',
    #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_68.tif',
    #                                                                       ]}
    # ## Patch 68
    output_dirs[68] = os.path.join(output_root, 'P_68_baseline_falsecolor')
    # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_densenet_augmented_fakelabels_tunedhyperparams')

    data_dirs[68] = os.path.join(data_root, 'Patch_ID_68_grn')
    #                    }
    input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg_sequ_16072020_transparent_reflectance_merged_green_nir_re_Patch_ID_68.tif']}

    # # Patch 90
    # output_dirs[90] = os.path.join(output_root, 'P_90_ssl')
    # data_dirs[90] = os.path.join(data_root, 'Patch_ID_90')
    # input_files_rgb[90] = {'pC_col_2020_plant_PS490_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_90.tif']}
    ## Patch 90
    output_dirs[90] = os.path.join(output_root, 'P_90_baseline_falsecolor')
    # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_densenet_augmented_fakelabels_tunedhyperparams')

    data_dirs[90] = os.path.join(data_root, 'Patch_ID_90_grn')
    #                    }
    input_files_rgb[90] = {'pC_col_2020_plant_PS490_Maiz_smc_Krig.tif': ['Tempelberg_sequ_16072020_transparent_reflectance_merged_green_nir_re_Patch_ID_90.tif']}


    ## Patch 65
    output_dirs[65] = os.path.join(output_root, 'P_65_densenet')

    data_dirs[65] = os.path.join(data_root, 'Patch_ID_65')
    # data_dirs[65] = os.path.join(data_root, 'Patch_ID_65_NDVI')

    input_files_rgb[65] = {'pC_col_2020_plant_PS465_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_65.tif']}
    # input_files_rgb[65] = {'pC_col_2020_plant_PS465_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_65.tif',
    #                                                                       'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_65.tif',
    #                                                                       ]}


    ## Patch 73
    output_dirs[73] = os.path.join(output_root, 'P_73_densenet')
    data_dirs[73] = os.path.join(data_root, 'Patch_ID_73')

    input_files[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
                                                                      'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
                       }
    input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']}

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

    ## Patch 58
    output_dirs[58] = os.path.join(output_root, 'P_58_densenet')
    data_dirs[58] = os.path.join(data_root, 'Patch_ID_58')

    input_files_rgb[58] = {'pC_col_2020_plant_PS458_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_58.tif']}
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

    ## Patch 76
    output_dirs[76] = os.path.join(output_root, 'P_76_densenet')
    data_dirs[76] = os.path.join(data_root, 'Patch_ID_76')
    input_files_rgb[76] = {'pC_col_2020_plant_PS476_Sun_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_76.tif']}

    ## Patch 81
    output_dirs[81] = os.path.join(output_root, 'P_81_baseline')
    data_dirs[81] = os.path.join(data_root, 'Patch_ID_81')
    input_files_rgb[81] = {'pC_col_2020_plant_PS481_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_81.tif']}


    ## HYPERPARAMETERS
    num_epochs = 200
    num_epochs_finetuning = 10
    lr = 0.001 # (Krizhevsky et al.2012)
    lr_finetuning = 0.0001
    momentum = 0.9 # (Krizhevsky et al.2012)
    wd = 0.0005 # (Krizhevsky et al.2012)
    classes = 1
    # batch_size = 16
    batch_size = 128  # (Krizhevsky et al.2012)
    num_folds = 1#9 # ranom-CV -> 1
    min_delta = 0.001
    patience = 10
    min_epochs = 200

    patch_no = 65
    test_patch_no = 58
    # test_patch_no = 95
    stride = 30 # 20 is too small
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

    this_output_dir = output_dirs[patch_no]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                     patch_id=patch_no,
                                     data_dir=data_dirs[patch_no],
                                     stride=stride,
                                     workers=os.cpu_count(),
                                     augmented=augmentation,
                                     input_features=features,
                                     batch_size=batch_size,
                                     validation_strategy=scv,
                                     fake_labels=fake_labels,
                                     input_type=input_type
                                     )
    test_datamodule = PatchCROPDataModule(input_files=input_files_rgb[test_patch_no],
                                          patch_id=test_patch_no,
                                          data_dir=data_dirs[test_patch_no],
                                          stride=stride,
                                          workers=os.cpu_count(),
                                          augmented=augmentation,
                                          input_features=features,
                                          batch_size=batch_size,
                                          validation_strategy=scv,
                                          fake_labels=fake_labels,
                                          input_type=input_type
                                          )

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

    if num_samples_per_fold == None:
        datamodule.prepare_data(num_samples=num_samples_per_fold)
        test_datamodule.prepare_data(num_samples=num_samples_per_fold)
    else:
        datamodule.load_subsamples(num_samples=num_samples_per_fold)
        test_datamodule.load_subsamples(num_samples=num_samples_per_fold)

    sets = ['train', 'val', 'test']
    # sets = ['val']
    global_pred = dict()
    global_y = dict()
    local_r2 = dict()
    global_r2 = dict()
    local_r = dict()
    global_r = dict()
    statistics_df = pd.DataFrame(columns=['Patch_ID', 'Features', 'Architecture', 'Set', 'Global_R2', 'Local_R2_F1', 'Local_R2_F2', 'Local_R2_F3', 'Local_R2_F4', 'Local_R2_F5', 'Local_R2_F6', 'Local_R2_F7', 'Local_R2_F8', 'Local_R2_F9'])
    # validation_set_pos= [[0, 0], [1162, 1162]]
    # validation_set_extend=[[1161, 1161], [1161, 1161]]
    validation_set_pos = [[1162, 0]]
    validation_set_extend = [[1161, 2323]]
    for s in sets:
        global_pred[s] = []
        global_y[s] = []
        local_r2[s] = []
        local_r[s] = []
        # i=0
        # for k in range(num_folds):
        for k in range(num_folds):

            # load data
            # datamodule.setup_fold_index(k)
            # test_datamodule.setup_fold(validation_set_pos=validation_set_pos[k],
            #                       validation_set_extend=validation_set_extend[k],
            #                       data_set_row_extend=2323,
            #                       data_set_column_extend=2323,
            #                       buffer=1,  # 223,
            #                       training_response_standardization=training_response_normalization
            #                       )
            datamodule.setup_fold(validation_set_pos=validation_set_pos[k],
                                  validation_set_extend=validation_set_extend[k],
                                  data_set_row_extend=2323,
                                  data_set_column_extend=2323,
                                  buffer= 1,#223,
                                  training_response_standardization=training_response_normalization
                                  )
            dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(), 'test': test_datamodule.all_dataloader()}
            # dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader(), 'test': datamodule.all_dataloader()}

            # set up model wrapper and input data
            if features == 'RGB':
                model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                                  device=device,
                                                  lr=lr,
                                                  momentum=momentum,
                                                  wd=wd,
                                                  pretrained=pretrained,
                                                  tune_fc_only=tune_fc_only,
                                                  model=architecture,
                                                  training_response_standardizer=datamodule.training_response_standardizer
                                                  )
            elif features == 'RGB+':
                model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=True, model=architecture, lr=lr, wd=wd)

            # load trained model weights with respect to possible parallelization and device
            if torch.cuda.is_available():
                state_dict = torch.load(checkpoint_paths[k])
            else:
                state_dict = torch.load(checkpoint_paths[k], map_location=torch.device('cpu'))

            # state_dict_revised = OrderedDict()
            # for key, value in state_dict.items():
            #     revised_key = key.replace("module.", "")
            #     state_dict_revised[revised_key] = value
            state_dict_revised = state_dict
            model_wrapper.model.load_state_dict(state_dict_revised)
            # else:
            #     model_wrapper.model.load_state_dict(state_dict, map_location=torch.device('cpu'))
            # # Parallelize model
            # if torch.cuda.device_count() > 1:
            # model_wrapper.model = nn.DataParallel(model_wrapper.model)

            # make prediction
            # for each fold store labels and predictions
            local_preds, local_labels = model_wrapper.predict(phase=s)

            # save labels and predictions for each fold
            torch.save(local_preds, os.path.join(this_output_dir, test_patch_name+'y_hat_' + s + '_' + str(k) + '.pt'))
            torch.save(local_labels, os.path.join(this_output_dir, test_patch_name+'y_' + s + '_' + str(k) + '.pt'))

            # for debugging, save labels and predictions in df
            y_yhat_df = pd.DataFrame({'y': local_labels, 'y_hat': local_preds})
            y_yhat_df.to_csv(os.path.join(this_output_dir, test_patch_name+'y-y_hat_{}_{}.csv'.format(s, k)), encoding='utf-8')

            # save local r2 for a certain train or val set
            local_r2[s].append(r2_score(local_labels, local_preds))
            local_r[s].append(np.corrcoef(local_labels, local_preds)[0][1])
            print('' + s + ':: {}-fold local R2 {}'.format(k, local_r2[s][k]))
            print('' + s + ':: {}-fold local r {}'.format(k, local_r[s][k]))
            # pool label and predictions for each set
            global_pred[s].extend(local_preds)
            global_y[s].extend(local_labels)
        global_r2[s] = r2_score(global_y[s], global_pred[s])
        global_r[s] = np.corrcoef(global_y[s], global_pred[s])[0][1]
        print('' + s + ':: global R2 {}'.format(global_r2))
        print('' + s + ':: global r {}'.format(global_r))
    #     # save predictions to CSV
    #     # statistics_df = pd.read_csv(os.path.join(output_root, 'Performance_all_Patches_RGB.csv'))
    #     statistics_df.loc[-1] = [patch_no, features, architecture, s, global_r2[s], local_r2[s][0], local_r2[s][1], local_r2[s][2], local_r2[s][3], local_r2[s][4], local_r2[s][5], local_r2[s][6], local_r2[s][7], local_r2[s][8]]
    #     statistics_df.index = statistics_df.index + 1
    #     statistics_df.sort_index()
    # statistics_df.to_csv(os.path.join(this_output_dir, 'prediction_statistics.csv'), encoding='utf-8')





    # if features == 'RGB':
    #     model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
    #                                       device=device,
    #                                       lr=lr,
    #                                       momentum=momentum,
    #                                       wd=wd,
    #                                       k=num_folds,
    #                                       pretrained=False,
    #                                       tune_fc_only=True,
    #                                       model=architecture,
    #                                       )
    # elif features == 'RGB+':
    #     model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=True, model=architecture, lr=lr, wd=wd)
    #
    # statistics_df = pd.DataFrame(columns=['Patch_ID', 'Features', 'Architecture', 'Set', 'Global_R2', 'Local_R2_F1', 'Local_R2_F2', 'Local_R2_F3', 'Local_R2_F4', 'Local_R2_F5', 'Local_R2_F6', 'Local_R2_F7', 'Local_R2_F8', 'Local_R2_F9'])
    # for s in sets:
    #     global_pred[s] = []
    #     global_y[s] = []
    #     local_r2[s] = []
    #     # i=0
    #     for k in range(num_folds):
    #         # ####### DEBUG
    #         # if k>0:
    #         #     break
    #         # load model and set to eval mode
    #         # model_wrapper.model.load_state_dict(torch.load(checkpoint_paths[k],map_location=torch.device('cpu')))
    #         model_wrapper.model.load_state_dict(torch.load(checkpoint_paths[k]))
    #         model_wrapper.model.to(device)
    #         model_wrapper.model.eval()
    #         # load data
    #         datamodule.setup_fold_index(k)
    #         if s == 'train':
    #             dl = datamodule.train_dataloader()
    #         elif s == 'val':
    #             dl = datamodule.val_dataloader()
    #
    #
    #         # for each fold store labels and predictions
    #         local_preds = []
    #         local_labels = []
    #
    #         # loop over dataset
    #         for inputs, labels in dl:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #             # make prediction
    #             with torch.no_grad():
    #                 y_hat = torch.flatten(model_wrapper.model(inputs))
    #             # while looping over the data loader save labels and predictions
    #             local_preds.extend(y_hat.detach().cpu().numpy())
    #             local_labels.extend(labels.detach().cpu().numpy())
    #
    #         # save labels and predictions for each fold
    #         torch.save(local_preds, os.path.join(this_output_dir, 'y_hat_' + s + '_' + str(k) + '.pt'))
    #         torch.save(local_labels, os.path.join(this_output_dir, 'y_' + s + '_' + str(k) + '.pt'))
    #
    #         # for debugging, save labels and predictions in df
    #         y_yhat_df = pd.DataFrame({'y':local_labels, 'y_hat':local_preds})
    #         y_yhat_df.to_csv(os.path.join(this_output_dir, 'y-y_hat_{}.csv'.format(k)), encoding='utf-8')
    #
    #         # save local r2 for a certain train or val set
    #         local_r2[s].append(r2_score(local_labels, local_preds))
    #         print(''+s+':: {}-fold local R2 {}'.format(k, local_r2[s][k]))
    #         # pool label and predictions for each set
    #         global_pred[s].extend(local_preds)
    #         global_y[s].extend(local_labels)
    #     global_r2[s] = r2_score(global_y[s], global_pred[s])
    #     print(''+s+':: global R2 {}'.format(global_r2))
    # #     # save predictions to CSV
    # #     # statistics_df = pd.read_csv(os.path.join(output_root, 'Performance_all_Patches_RGB.csv'))
    # #     statistics_df.loc[-1] = [patch_no, features, architecture, s, global_r2[s], local_r2[s][0], local_r2[s][1], local_r2[s][2], local_r2[s][3], local_r2[s][4], local_r2[s][5], local_r2[s][6], local_r2[s][7], local_r2[s][8]]
    # #     statistics_df.index = statistics_df.index + 1
    # #     statistics_df.sort_index()
    # # statistics_df.to_csv(os.path.join(this_output_dir, 'prediction_statistics.csv'), encoding='utf-8')