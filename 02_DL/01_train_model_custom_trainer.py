# -*- coding: utf-8 -*-
"""
script to train model
"""

# Built-in/Generic Imports
import time
import os
from copy import deepcopy

# Libs
from pytorch_lightning import seed_everything, Trainer
from sklearn.metrics import r2_score
import numpy as np
import  pandas as pd

import torch
from torch import nn
import torchvision.models as models

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

# Own modules
from PatchCROPDataModule import PatchCROPDataModule
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


def train_model(model, dataloaders, criterion, optimizer, device, patience:int =10, delta:float =0, num_epochs=25):
    since = time.time()
    val_mse_history = []
    train_mse_history = []

    # mse = MeanSquaredError()

    best_model_wts = deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if patience == 0:
            break
        epoch_loss = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(torch.flatten(outputs), labels.data)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train' and epoch_loss < best_loss:
            # if phase == 'val' and epoch_loss < best_loss:
                print('saving best model')
                best_loss = epoch_loss
                best_model_wts = deepcopy(model.state_dict())
                best_epoch = epoch
                patience = 10
            if phase == 'val' and (epoch_loss-best_loss) > delta:#epoch_loss > best_loss:
                print('patience: {}'.format(patience))
                patience -= 1
            if phase == 'val':
                val_mse_history.append(epoch_loss)
            if phase == 'train':
                train_mse_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val MSE: {:4f}'.format(best_mse))
    print('Best val MSE: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_mse_history, train_mse_history, best_epoch

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
    output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_not_augmented_custom_origdata')
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
    lr = 0.01
    momentum = 0.8
    wd = 0.01
    classes = 1
    batch_size = 20
    num_folds = 9

    patch_no = 119
    architecture = 'densenet'
    # architecture = 'resnet50'
    augmentation = True
    tune_fc_only = False
    features = 'RGB'
    # features = 'RGB+'
    num_samples = None

    this_output_dir = output_dirs[patch_no]

    print('Setting up data in {}'.format(data_dirs[patch_no]))
    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no], data_dir=data_dirs[patch_no], stride=10, workers=os.cpu_count(), augmented=augmentation)
    datamodule.prepare_data()

    # loop over folds, last fold is for testing only
    for k in range(num_folds):
        print(f"STARTING FOLD {k}")
        # Detect if we have a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('working on device %s' % device)

        if features == 'RGB':
            model_wrapper = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture, lr=lr, wd=wd)
        elif features == 'RGB+':
            model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture, lr=lr, wd=wd)

        datamodule.setup_fold_index(k)
        dataloaders_dict = {'train': datamodule.train_dataloader(num_samples=num_samples), 'val': datamodule.val_dataloader(num_samples=num_samples)}

        # Send the model to GPU
        model_wrapper.model.to(device)

        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        # Train and evaluate
        print('training for {} epochs'.format(num_epochs))
        best_model, val_losses, train_losses, best_epoch = train_model(model_wrapper.model, dataloaders_dict, loss_fn, optimizer, delta=0.1, device=device, num_epochs=num_epochs, )
        # save best model
        torch.save(best_model.state_dict(), os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt'))
        # save training statistics
        df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses, 'best_epoch': best_epoch})
        df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) + '.csv'), encoding='utf-8')

        load_model = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
        load_model.model.load_state_dict(torch.load(os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt')))
        load_model.model.to(device)
        load_model.model.eval()

        compare_models(best_model, load_model.model)

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
