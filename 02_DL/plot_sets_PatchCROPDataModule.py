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
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision.models as models

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision.transforms as transforms

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

    # data_root = '/beegfs/stiller/PatchCROP_all/Data/'
    data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
    # output_root = '/beegfs/stiller/PatchCROP_all/Output/'
    output_root = '../../Output/'

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
    output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densened_augmented_custom_btf_art_labels_hyper')
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
    num_epochs = 10
    lr = 0.001
    momentum = 0.8
    wd = 0.01
    classes = 1
    batch_size = 1
    num_folds = 1

    patch_no = 73
    stride= 20
    architecture = 'densenet'
    # architecture = 'resnet50'
    augmentation = True
    tune_fc_only = True
    features = 'RGB'
    # features = 'RGB+'
    num_samples_per_fold = None

    this_output_dir = output_dirs[patch_no]

    datamodule = PatchCROPDataModule(input_files=input_files_rgb[patch_no], patch_id=patch_no,
                                     data_dir=data_dirs[patch_no], stride=stride, workers=os.cpu_count(),
                                     augmented=augmentation, input_features=features, batch_size=batch_size)
    datamodule.prepare_data(num_samples=num_samples_per_fold)
    # datamodule.load_subsamples(num_samples=num_samples_per_fold)

    if features == 'RGB':
        mean_norm = torch.tensor([0.485, 0.456, 0.406])
        std_norm = torch.tensor([0.229, 0.224, 0.225])
    elif features == 'RGB+':
        mean = torch.tensor([0.485, 0.456, 0.406, 0])
        std = torch.tensor([0.229, 0.224, 0.225, 1])
    # loop over folds, last fold is for testing only
    k_sets = []
    train_sets = []
    for k in range(num_folds):
        print('-'*20)
        print(f"STARTING FOLD {k}")
        # # Detect if we have a GPU available
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('working on device %s' % device)
        #
        # if features == 'RGB':
        #     model_wrapper = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
        # elif features == 'RGB+':
        #     model_wrapper = MCYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)

        datamodule.setup_fold_index(k)
        dl = datamodule.val_dataloader()
        k_set = []
        i=0
        for inputs, labels in dl:
            k_set.extend(labels.numpy())
            if i<8:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(inputs.squeeze().permute(1,2,0)*std_norm+mean_norm)
                plt.savefig(os.path.join(this_output_dir, 'input_val_f{}_{}.png'.format(k,i)))
                i+=1
        # k_sets.append(datamodule.splits[k].datasets[-1].tensors[1].numpy())
        k_sets.append(k_set)
        print('Test set mean: {}'.format(np.mean(k_set)))
        print('Test set SD: {}'.format(np.std(k_set)))
        print('Test set Max: {}'.format(max(k_set)))



        # dataloaders_dict = {'train': datamodule.train_dataloader(), 'val': datamodule.val_dataloader()}

        dl = datamodule.train_dataloader()
        train_labels = []
        i=0
        for inputs, labels in dl:
            train_labels.extend(labels.numpy())
            if i<8:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(inputs.squeeze().permute(1,2,0)*std_norm+mean_norm)
                plt.savefig(os.path.join(this_output_dir, 'input_train_f{}_{}.png'.format(k,i)))
                i+=1

        train_sets.append(train_labels)
        print('Train set mean: {}'.format(np.mean(train_labels)))
        print('Train set SD: {}'.format(np.std(train_labels)))
        print('Train set Max: {}'.format(max(train_labels)))

    all_labels_val = []
    all_k = []
    for k in range(num_folds):
        all_labels_val.extend(k_sets[k])
        all_k.extend(np.ones(len(k_sets[0]))*k)
    df_dict = {'l': all_labels_val, 'k': all_k}
    df = pd.DataFrame(df_dict)

    fig = plt.figure()
    ax = sns.violinplot(x="k", y="l", data=df)
    fig.savefig(os.path.join(this_output_dir, 'val_folds_dist_{}.png'.format(num_samples_per_fold)))

    all_labels_train = []
    all_k = []
    for k in range(num_folds):
        all_labels_train.extend(train_sets[k])
        all_k.extend(np.ones(len(train_sets[k])) * k)
    df_dict = {'l': all_labels_train, 'k': all_k}
    df = pd.DataFrame(df_dict)

    fig = plt.figure()
    ax = sns.violinplot(x="k", y="l", data=df)
    fig.savefig(os.path.join(this_output_dir, 'train_folds_dist_{}.png'.format(num_samples_per_fold)))

        # # Send the model to GPU
        # model_wrapper.model.to(device)
        #
        # loss_fn = nn.MSELoss(reduction='mean')
        # optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        # # Train and evaluate
        # print('training for {} epochs'.format(num_epochs))
        # best_model, val_losses, train_losses, best_epoch = train_model(model_wrapper.model, dataloaders_dict, loss_fn, optimizer, delta=0.1, device=device, num_epochs=num_epochs, )
        # # save best model
        # torch.save(best_model.state_dict(), os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt'))
        # # save training statistics
        # df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses, 'best_epoch': best_epoch})
        # df.to_csv(os.path.join(this_output_dir, 'training_statistics_f' + str(k) + '.csv'), encoding='utf-8')
        #
        # load_model = RGBYieldRegressor(pretrained=True, tune_fc_only=tune_fc_only, model=architecture)
        # load_model.model.load_state_dict(torch.load(os.path.join(this_output_dir, 'model_f' + str(k) + '.ckpt')))
        # load_model.model.to(device)
        # load_model.model.eval()
        #
        # compare_models(best_model, load_model.model)
