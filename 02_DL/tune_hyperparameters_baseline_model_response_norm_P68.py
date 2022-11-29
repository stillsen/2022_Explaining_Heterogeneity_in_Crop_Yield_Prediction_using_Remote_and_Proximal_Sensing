# -*- coding: utf-8 -*-
"""
Wrapper class for crop yield regression model employing a ResNet50 or densenet in PyTorch in a transfer learning approach using
RGB remote sensing observations
"""

# Built-in/Generic Imports

# Libs
import time
from copy import deepcopy


from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.util import inspect_serializability


import warnings
import random
import numpy as np
import os

import torch
import torchvision.models as models
import torch.nn as nn

# Own modules
from RGBYieldRegressor import RGBYieldRegressor
from PatchCROPDataModule import PatchCROPDataModule
from MC_YieldRegressor import MCYieldRegressor

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'




class TuneYieldRegressor(tune.Trainable):
    def setup(self, config):
        warnings.warn('hyperparameter space needs proper setup', FutureWarning)

        # seed_everything(42)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        output_dirs = dict()
        data_dirs = dict()
        input_files = dict()
        input_files_rgb = dict()

        data_root = '/beegfs/stiller/PatchCROP_all/Data/'
        # data_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
        output_root = '/beegfs/stiller/PatchCROP_all/Output/'
        # output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

        ## Patch 68
        output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_baselinemodel_hyperparams_all_response_norm')
        # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_baselinemodel_hyperparams_all')
        # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_densenet_augmented_fakelabels_fixhyperparams')
        # output_dirs[68] = os.path.join(output_root, 'Patch_ID_68_RGB_densenet_augmented_fakelabels_tunedhyperparams')

        data_dirs[68] = os.path.join(data_root, 'Patch_ID_68')
        # data_dirs[68] = os.path.join(data_root, 'Patch_ID_68_NDVI')

        # input_files[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
        #                                                                   'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
        #                                                                   'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
        #                                                                   'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
        #                    }
        input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif']}
        # input_files_rgb[73] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif',
        #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_68.tif',
        #                                                                       ]}

        ## Patch 73
        # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_fixhyperparams')
        # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baseline_augmented_fakelabels_raytuning_lr')
        # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baseline_augmented_fakelabels_raytuning_wd')
        output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baseline_augmented_raytuning_lr')
        # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baselinemodel_augmented_raytuning')
        # output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_augmented_fakelabels_fixhyperparams')
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

        ## HYPERPARAMETERS
        num_epochs = 100
        num_epochs_finetuning = 10
        # lr = 0.001 # (Krizhevsky et al.2012)
        lr = 0.0001 # tuning 1
        lr_finetuning = 0.0005
        momentum = 0.9 # (Krizhevsky et al.2012)
        # wd = 0.0005 # (Krizhevsky et al.2012)
        wd = 0.001 # tuning 1
        classes = 1
        batch_size = 128 #(Krizhevsky et al.2012)
        # batch_size = 256 # tuning 1
        num_folds = 1#9 # ranom-CV -> 1
        min_delta = 0.01 # aka 1%
        patience = 10
        min_epochs = 100

        # patch_no = 73
        patch_no = 68
        stride = 30 # 20 is too small
        architecture = 'baselinemodel'
        # architecture = 'densenet'
        # architecture = 'resnet50'
        augmentation = True
        tune_fc_only = False
        pretrained = False
        features = 'RGB'
        # features = 'RGB+'
        num_samples_per_fold = None # subsamples? -> None means do not subsample but take whole fold
        scv = True
        # scv = False
        fake_labels = False
        training_response_normalization = True
        # training_response_normalization = False

        this_output_dir = output_dirs[patch_no]

        print('Setting up data in {}'.format(data_dirs[patch_no]))
        datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no],
                                         patch_id=patch_no,
                                         data_dir=data_dirs[patch_no],
                                         stride=stride,
                                         workers=os.cpu_count(),
                                         augmented=augmentation,
                                         input_features=features,
                                         # batch_size=batch_size,
                                         batch_size=config['batch_size'],
                                         scv=scv,
                                         fake_labels=fake_labels,
                                         )
        datamodule.prepare_data(num_samples=num_samples_per_fold)

        print('working on patch {}'.format(patch_no))
        # loop over folds, last fold is for testing only

        # Detect if we have a GPU available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda")
        print('working on device %s' % self.device)

        # ############################### DEBUG
        # warnings.warn('training on 1 fold', FutureWarning)
        # validation_set_pos= [[0, 0], [1162, 1162]]
        # validation_set_extend=[[1161, 1161], [1161, 1161]]
        validation_set_pos = [[1162, 0]]
        validation_set_extend = [[1161, 2323]]
        for k in range(1):
        # for k in range(num_folds):
            # print(f"STARTING FOLD {k}")


            # data
            # datamodule.setup_fold_index(k)
            datamodule.setup_fold(validation_set_pos=validation_set_pos[k],
                                  validation_set_extend=validation_set_extend[k],
                                  data_set_row_extend=2323,
                                  data_set_column_extend=2323,
                                  buffer = 1,#223,
                                  training_response_standardization=training_response_normalization
                                  )

            dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader()}

            #### TUNE LAST LAYER, FREEZE BEFORE::
            if features == 'RGB':
                self.model_wrapper = RGBYieldRegressor(dataloaders=dataloaders_dict,
                                                  device=self.device,
                                                  # lr=lr,
                                                  lr=config['lr'],
                                                  momentum=momentum, #config['momentum'],
                                                  wd=config['wd'],
                                                  # wd=wd,
                                                  # k=num_folds,
                                                  pretrained=pretrained,
                                                  tune_fc_only=tune_fc_only,
                                                  model=architecture,
                                                  training_response_standardizer=datamodule.training_response_standardizer
                                                       )
            # elif features == 'RGB+':
            #     model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=True, model=architecture, lr=config['lr'], wd=['wd'])

            # Send the model to GPU
            if torch.cuda.device_count() > 1:
                self.model_wrapper.model = nn.DataParallel(self.model_wrapper.model)
            self.model_wrapper.model.to(self.device)

            # # warnings.warn('training missing', FutureWarning)
            # # Train and evaluate
            # print('training for {} epochs'.format(num_epochs))
            # model_wrapper.train_model(patience=5,
            #                           min_delta=0.0,
            #                           num_epochs=num_epochs,
            #                           )
        # use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.train_loader = dataloaders_dict['train']
        # self.test_loader = dataloaders_dict['val']
        # self.model = model_wrapper.model
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=config['lr'],
        #     momentum=config['momentum'],
        #     weight_decay=config['wd']
        # )


    def step(self):
        train_loss = self.model_wrapper.train()
        val_loss = self.model_wrapper.test()
        return {"train_loss": train_loss, "val_loss": val_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        torch.save(self.model_wrapper.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "tuning_model.pth")
        self.model_wrapper.model.load_state_dict(torch.load(checkpoint_path))

if __name__ == "__main__":
    param_space = {
        "lr": tune.loguniform(1e-6, 1e-1),
        # "lr": tune.grid_search([5*1e-3, 1e-3, 1e-4]),
        # "momentum": tune.uniform(0.7, 0.99),
        "wd": tune.grid_search([5*1e-3, 1e-3, 5*1e-4, 1e-4, 5*1e-5, 1e-5, 0]),
        "batch_size": tune.grid_search([8, 16, 32, 64, 128, 256, 512]),
    }

    hyperband = ASHAScheduler(metric="val_loss", mode="min")

    analysis = tune.run(TuneYieldRegressor,
                        checkpoint_freq=10,
                        max_failures=5,
                        stop={"training_iteration" : 20},
                        config=param_space,
                        resources_per_trial={"gpu":2},
                        metric='val_loss',
                        mode='min',
                        # num_samples=20,
                        # scheduler=hyperband,
                        resume="AUTO",
                        name='YieldRegressor_baseline_all_response_norm',
                        # name='YieldRegressor_baseline_all',
                        )
    print('best config: ', analysis.get_best_config(metric="train_loss", mode="min"))
    print('best config: ', analysis.get_best_config(metric="val_loss", mode="min"))

    # output_dir = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/Patch_ID_73_RGB_densenet_augmented_custom_btf_art_labels_ray'
    output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_68_RGB_baselinemodel_hyperparams_all_response_norm'
    # output_dir = '/beegfs/stiller/PatchCROP_all/Output/P_68_RGB_baselinemodel_hyperparams_all'
    # output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_1'
    torch.save(analysis, os.path.join(output_dir, 'analysis.ray'))