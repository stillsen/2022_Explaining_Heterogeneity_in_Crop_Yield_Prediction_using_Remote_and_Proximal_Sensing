# -*- coding: utf-8 -*-
"""
PyTorch Lightning Data Module Class
that works on files in an input folder, i.e. analysis folder f under
../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC .

For patch i:
1) load each corresponding filename_Patch_ID_XXX.tif file in that folder and upsample the resolution
   to 2977px x 2977px for no flower strip patches and to 2977px x 2480px with flower strips
2) normalize  all raster files, i.e. bands as well as DSM an all other
3) combine all m raster files with v_{i} bands each to a m*v_{i} tensor
4) spatially divide this tensor into k folds (for spatial cross validation)
5) generate samples in each fold of size 224px x 224px by sliding a window over the fold image with step size v;
   (no transform here)
6) OPTION: if multiple patches are in folder f, perform steps 1 to 4 for all patches and stack the folds
7) export each fold as pytorch tensor as .pt

output: Foldername.pt
"""

# Built-in/Generic Imports
import os

# Libs
from osgeo import gdal
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np

import os.path as osp
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from torch.utils.data import Subset

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning.loops.base import Loop
# from pytorch_lightning.loops.fit_loop import FitLoop
# from pytorch_lightning.trainer.states import TrainerFn
# from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
# from pytorch_lightning.callbacks import Callback, EarlyStopping
#
# from torchmetrics import R2Score
# from ignite.contrib.metrics.regression import R2Score
# from ignite.engine import *

# Own modules


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'



#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################


class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class PatchCROPDataModule:

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def __init__(self, input_files: dict, patch_id, data_dir: str = './', batch_size: int = 20, stride:int = 10, workers:int = 0, input_features:str = 'RGB', augmented:bool = False):
        super().__init__()
        self.flowerstrip_patch_ids = ['12', '13', '19', '20', '21', '105', '110', '114', '115', '119']
        self.patch_ids = ['12', '13', '19', '20', '21', '39', '40', '49', '50', '51', '58', '59', '60', '65', '66', '68', '73', '74', '76', '81', '89', '90', '95', '96', '102', '105', '110', '114', '115', '119']

        self.data_dir = data_dir
        # name of the label file
        self.input_files = input_files
        self.num_folds = 9
        self.batch_size = batch_size
        self.stride = stride
        self.workers = workers
        self.augmented = augmented
        self.patch_id = patch_id
        self.input_features = input_features

        # Datesets
        self.splits = [[] for i in range(self.num_folds)]
        # 0-8 rotate for train and val; 8 is test set
        self.split_idx = [split for split in KFold(self.num_folds).split(range(self.num_folds))]
        self.setup_done = False


        # self.prepare_data_per_node=False
        # # multi spectral setting
        # # max and min values for multi spectral bands
        # self.ms_band_ranges = {'green': (0.53, 0.57),
        #                        'red': (0.64, 0.68),
        #                        'nir': (0.77, 0.81),
        #                        'rededge': (0.73, 0.74),
        #                        'dsm': (0, 1),
        #                        'ndvi': (-1, 1)}
        # self.ms_band_minmax = {'11062020':{'green': (0.53, 0.57),
        #                                     'red': (0.64, 0.68),
        #                                     'nir': (0.77, 0.81),
        #                                     'rededge': (0.73, 0.74)},
        #                        '16072020':{'green': (0.53, 0.57),
        #                                     'red': (0.64, 0.68),
        #                                     'nir': (0.77, 0.81),
        #                                     'rededge': (0.73, 0.74)},
        #                        '17062020':{'green': (0.53, 0.57),
        #                                     'red': (0.64, 0.68),
        #                                     'nir': (0.77, 0.81),
        #                                     'rededge': (0.73, 0.74)}}
        # self.ms_band_order = ['green', 'nir', 'red', 'rededge']
    def prepare_data(self, num_samples:int = None) -> None: # would actually make more sense to put it to setup, but there it is called as several processes
        # load the data
        # normalize
        # split
        # oversample
        # load over feature label combinations
        # if self.augmented:
        #     kfold_dir = os.path.join(self.data_dir,'kfold_set_augmented')
        # else:
        #     kfold_dir = os.path.join(self.data_dir, 'kfold_set')
        kfold_dir = os.path.join(self.data_dir, 'kfold_set')
        if not self.setup_done:
            # check if already build and load
            if os.path.isdir(kfold_dir):
                # load
                print('loading data')
                for k in range(self.num_folds):
                    if self.input_features == "RGB":
                        file_name = 'Patch_ID_' + str(self.patch_id) + '_fold_' + str(k) + '.pt'
                    elif self.input_features == "RGB+":
                        file_name = 'Patch_ID_' + str(self.patch_id) + '_NDVI_fold_' + str(k) + '.pt'
                    elif self.input_features == "MSI":
                        file_name = 'Patch_ID_' + str(self.patch_id) + '_MSI_fold_' + str(k) + '.pt'

                    # f = torch.load(os.path.join(kfold_dir,file), map_location=torch.device('cuda'))
                    f = torch.load(os.path.join(kfold_dir, file_name))
                    self.splits[int(k)] = f
                # for file in os.listdir(kfold_dir):
                #     k = file.split('.')[0][-1]
                #     # f = torch.load(os.path.join(kfold_dir,file), map_location=torch.device('cuda'))
                #     f = torch.load(os.path.join(kfold_dir, file))
                #     self.splits[int(k)] = f
            else:
                # otherwise build oversampled k-fold set
                print('generating data')
                for label, features in self.input_files.items():
                    label_matrix = torch.tensor([])
                    feature_matrix = torch.tensor([])
                    # load label
                    label_matrix = torch.tensor(gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray())
                    # label_matrix = gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray()
                    # load and concat features
                    for idx, feature in enumerate(features):
                        # if RGB drop alpha channel, else all
                        if 'soda' in feature or 'Soda' in feature: # RGB
                            print(feature)
                            f = torch.tensor(gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray()[:3,:,:]) # only first 3 channels -> not alpha channel
                        else: # multichannel
                            f = torch.tensor(gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray())
                        # concat feature tensor, e.g. RGB tensor and NDVI tensor
                        if f.dim() == 2:
                            feature_matrix = torch.cat((feature_matrix, f.unsqueeze(0)), 0)
                        else:
                            feature_matrix = torch.cat((feature_matrix, f), 0)

                    # spatial splits for k fold spatial cross validation and call sliding window  -> 224x224 images
                    self.setup_folds(label_matrix, feature_matrix)
                self.splits = [ConcatDataset(ensemble) for ensemble in self.splits]
                # save splits
                os.mkdir(kfold_dir)
                for idx in range(self.num_folds):
                    file_name = kfold_dir+'/'+self.data_dir.split('/')[-1]+'_fold_'+str(idx)+'.pt'
                    torch.save(self.splits[idx], file_name)
            self.setup_done = True
        if num_samples != None:
            idx = random.sample(range(3025), num_samples)
            self.splits = [Subset(split, idx) for split in self.splits]
            for idx in range(self.num_folds):
                    file_name = kfold_dir+'/'+self.data_dir.split('/')[-1]+'_subsample_'+str(num_samples)+'_fold_'+str(idx)+'.pt'
                    torch.save(self.splits[idx], file_name)

    def load_subsamples(self, num_samples:int=None):
        if num_samples != None:
            kfold_dir = os.path.join(self.data_dir, 'kfold_set')
            # check if already build and load
            if os.path.isdir(kfold_dir):
                # load
                for k in range(self.num_folds):
                    file_name = 'Patch_ID_'+str(self.patch_id)+'_subsample_'+str(num_samples)+'_fold_'+str(k)+'.pt'
                    # f = torch.load(os.path.join(kfold_dir,file), map_location=torch.device('cuda'))
                    f = torch.load(os.path.join(kfold_dir, file_name))
                    self.splits[int(k)] = f
        else:
            raise ValueError('number of samples need to be > 1.')

    def setup_folds(self, label_matrix, feature_matrix, lower_bound: int =327, kernel_size: int =224) -> None:
        '''
        :param label_matrix: kriging interpolated yield maps
        :param feature_matrix: stack of feature maps, such as rgb remote sensing, multi-channel remote sensing or dsm or ndvi
        both are required to have same dimension, but have different size according to whether the patch is a flower strip
        or not
        :return: constructs self.splits containing images (C H W) but in range [0 ... 255]
        whereas RGB format would be (H W C) range [0 ... 255]
        pytorch pretraining exptexted (C H W) [0...1]
        '''

        fold_starts_x = [0, 774, 1548]
        fold_length_x = 774
        if feature_matrix.shape[1] == 2977: # -> no flower strip patch
            fold_starts_y = [0, 774, 1548]
            fold_length_y = 774
        else: # flower strip patch
            fold_starts_y = [0, 645, 1290]
            fold_length_y = 645

        ############### DEBUG INSERT
        fold_starts_x = [0, 0, 0]
        fold_starts_y = [0, 0, 0]
        ###############

        upper_bound_x = feature_matrix.shape[2] - lower_bound
        upper_bound_y = feature_matrix.shape[1] - lower_bound

        # clip to tightest inner rectangle
        # and normalize feature array to range [0,1]
        # this could be outsourced to transforms.toTensor(), however this would require the input to be a PIL image
        feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]

        # normalize non-rgb channels to a range [0...255]
        # s.t. they can later be transformed to work with pre-training initialized weights
        if self.input_features == 'RGB' or self.input_features == 'RGB+':
            feature_arr[:3, :, :] = (feature_arr[:3, :, :] ) / 255
        if self.input_features == 'RGB+':
            # raise NotImplementedError('NDVI support for multi-modal multi-source input not yet supported')
            if feature_arr.shape[0] > 3: # normalize rest of channels, i.e. NDVI
                    # feature_arr[3:, :, :] = feature_matrix[3:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]
                    # feature_arr_min = (feature_arr.min(1, keepdim=True)[0]).min(2, keepdim=True)[0]
                    # feature_arr_max = (feature_arr.max(1, keepdim=True)[0]).max(2, keepdim=True)[0]
                    # feature_arr[3, :, :] -= torch.flatten(feature_arr_min)[3:]
                    # feature_arr[3, :, :] /= (torch.flatten(feature_arr_max)[3:]-torch.flatten(feature_arr_min)[3:])

                    feature_arr[3, :, :] = (feature_arr[3, :, :] + 1) / 2
            else:
                raise ValueError("No NDVI channel found")
        elif self.input_features == 'MSI':
            # raise NotImplementedError("torch version, NumPy missing")
            ##### torch version
            feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]
            feature_arr_min = feature_arr.view(feature_arr.size(0),-1).min(1,keepdim=True)[0] # get channel-wise min
            feature_arr_max = feature_arr.view(feature_arr.size(0), -1).max(1, keepdim=True)[0] # get channel-wise max
            feature_arr -= feature_arr_min.unsqueeze(1) # channel-wise substraction
            feature_arr /= feature_arr_max.unsqueeze(1)

        ########  --> transforms ----> NOPEEEEEEE, otherwise one hell of type conversion messes
        # if feature_RGB:
        #     feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]
        #     feature_arr[:3, :, :] /= 255 # normalzie RGB channels
        #     if feature_arr.shape[0] > 3: # normalize rest of channels, i.e. NDVI
        #         # feature_arr[3:, :, :] = feature_matrix[3:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]
        #         feature_arr_min = (feature_arr.min(1, keepdim=True)[0]).min(2, keepdim=True)[0]
        #         feature_arr_max = (feature_arr.max(1, keepdim=True)[0]).max(2, keepdim=True)[0]
        #         feature_arr[3, :, :] -= torch.flatten(feature_arr_min)[3:]
        #         feature_arr[3, :, :] /= (torch.flatten(feature_arr_max)[3:]-torch.flatten(feature_arr_min)[3:])
        #         # feature_arr_min = feature_arr.view(feature_arr.size(0), -1).min(1, keepdim=True)[0]  # get channel-wise min
        #         # feature_arr_max = feature_arr.view(feature_arr.size(0), -1).max(1, keepdim=True)[0]  # get channel-wise max
        #         # feature_arr[3:, :, :] -= feature_arr_min.unsqueeze(1)  # channel-wise substraction
        #         # feature_arr[3:, :, :] /= feature_arr_max.unsqueeze(1)
        # else:
        #     feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]
        #     feature_arr_min = feature_arr.view(feature_arr.size(0),-1).min(1,keepdim=True)[0] # get channel-wise min
        #     feature_arr_max = feature_arr.view(feature_arr.size(0), -1).max(1, keepdim=True)[0] # get channel-wise max
        #     feature_arr -= feature_arr_min.unsqueeze(1) # channel-wise substraction
        #     feature_arr /= feature_arr_max.unsqueeze(1)

        ############### DEBUG INSERT
        label_arr = (feature_arr[0,:,:] + feature_arr[1,:,:] + feature_arr[2,:,:])/3
        # label_arr = feature_arr[1, :, :]
        # label_arr = torch.max(feature_arr,dim=0)[0]
        ############### DEBUG DISABLE
        # label_arr = label_matrix[lower_bound:upper_bound_y, lower_bound:upper_bound_x] #label matrix un-normalized

        counter = 0
        for fold_x in fold_starts_x:
            for fold_y in fold_starts_y:
                print('fold: {}, x: {}, y: {}'.format(counter, fold_x, fold_y))

                feature_fold_arr = feature_arr[:, fold_y:fold_y + fold_length_y, fold_x:fold_x + fold_length_x]
                label_fold_arr = label_arr[fold_y:fold_y + fold_length_y, fold_x:fold_x + fold_length_x]

                # calculating start positions of kernels
                # x and y are need to able to take different values, as patches with flowerstrips are smaller
                x_size = feature_fold_arr.shape[2]
                y_size = feature_fold_arr.shape[1]
                possible_shifts_x = int((x_size - kernel_size) / self.stride)
                x_starts = np.array(list(range(possible_shifts_x))) * self.stride
                possible_shifts_y = int((y_size - kernel_size) / self.stride)
                y_starts = np.array(list(range(possible_shifts_y))) * self.stride

                # loop over start postitions and save kernel as separate image
                feature_tensor = None
                label_tensor = None
                for y in y_starts:
                    if y == y_starts[int(len(y_starts) / 4)]:
                        print("........... 25%")
                    if y == y_starts[int(len(y_starts) / 2)]:
                        print("........... 50%")
                    if y == y_starts[int(len(y_starts) * 3 / 4)]:
                        print("........... 75%")
                    for x in x_starts:
                        # shift kernel over image and extract kernel part
                        # only take RGB value
                        feature_kernel_img = feature_fold_arr[:, y:y + kernel_size, x:x + kernel_size]
                        label_kernel_img = label_fold_arr[y:y + kernel_size, x:x + kernel_size]
                        if x==0 and y==0:
                            feature_tensor = feature_kernel_img.unsqueeze(0)
                            label_tensor = label_kernel_img.mean().unsqueeze(0)
                        else:
                            feature_tensor = torch.cat((feature_tensor,feature_kernel_img.unsqueeze(0)), 0)
                            label_tensor = torch.cat((label_tensor, label_kernel_img.mean().unsqueeze(0)), 0)
                label_tensor = label_tensor.type(torch.HalfTensor)
                feature_tensor = feature_tensor.type(torch.HalfTensor)
                self.splits[counter].append((feature_tensor,label_tensor))
                counter += 1

    def setup_fold_index(self, fold_index: int) -> None:
        '''
        Sets up validation and train data set and applies normalization for both and augmentations for the train set if
        self.augmented.
        :param fold_index: index of the validation set, folds with i != fold_index are combined to train set
        :return: None
        '''
        # get indices
        train_indices, val_indices = self.split_idx[fold_index]

        if self.input_features == 'RGB':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        elif self.input_features == 'RGB+':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0],
                                             std=[0.229, 0.224, 0.225, 1])


        # set up train and validation data set and apply respective transformations and/or augmentations
        # given by pytorch and ImageNet pretraining
        transformer = transforms.Compose([#transforms.ToPILImage(),
                                          #transforms.ToTensor(),
                                          normalize,
                                          ])

        self.val_fold = ConcatDataset([TransformTensorDataset(self.splits[i],transform=transformer) for i in val_indices])

        # only train set would be augmented if enabled
        if self.augmented:
            transformer = transforms.Compose([#transforms.ToPILImage(),
                                              #transforms.ToTensor(),
                                              normalize,
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              ])

        self.train_fold = ConcatDataset([TransformTensorDataset(self.splits[i], transform=transformer) for i in train_indices])



    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, batch_size=self.batch_size, num_workers=self.workers)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, num_workers=self.workers, batch_size=self.batch_size)

    # def test_dataloader(self) -> DataLoader:
    #     # transformer = transforms.Compose([transforms.ToTensor()])
    #     # return transformer(DataLoader(self.test_dataset))
    #     return DataLoader(self.test_dataset, num_workers=self.workers, batch_size=self.batch_size)

    # def predict_dataloader(self) -> DataLoader:
    #     # transformer = transforms.Compose([transforms.ToTensor()])
    #     # return transformer(DataLoader(self.test_dataset))
    #     return DataLoader(self.test_dataset, num_workers=self.workers, batch_size=self.batch_size)

    def __post_init__(cls):
        super().__init__()



#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################


# class SpatialCVModel(LightningModule):
#     def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
#         super().__init__()
#         # Create `num_folds` models with their associated fold weights
#         self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
#         # self.predictions = torch.tensor([])
#         # self.y = torch.tensor([])
#         self.global_r2 = R2Score()
#         self.local_r2 = [R2Score() for i in range(8)]
#
#     def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
#         # Compute predictions over the `num_folds` models.
#         # report global and local r2 score
#         y = batch[1]
#         y_hat = torch.stack([m(batch[0]) for m in self.models])
#
#         # self.log("y is cuda", y.is_cuda, logger=True)
#         # self.log("y_hat is cuda", y_hat[0,:].squeeze().is_cuda, logger=True)
#
#         # y = y.to(device=self.device)
#         # y_hat = y_hat.to(device=self.device)
#
#         # print("{}, {}".format(y_hat[1,:].squeeze().shape, y_hat.shape))
#         for i in range(8):
#             this_y_hat = y_hat[i,:].squeeze()
#             # print(y.is_cuda)
#             # print(this_y_hat.is_cuda)
#             self.local_r2[i].update(this_y_hat, y)
#             self.log("local R2 - M{}".format(i), self.local_r2[i].compute(), prog_bar=True, logger=True)
#
#         self.global_r2.update(y_hat.flatten(), y.repeat(8))
#         self.log("global R2", self.global_r2.compute(), prog_bar=True, logger=True)
#

#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################
#
# class PrintCallback(Callback):
#     def on_train_start(self, trainer, pl_module):
#         print("Training is started!")
#     def on_train_end(self, trainer, pl_module):
#         print("Training is done.")
#
#
# class KFoldLoop(Loop):
#     def __init__(self, num_folds: int, export_path: str) -> None:
#         super().__init__()
#         self.num_folds = num_folds
#         self.current_fold: int = 0
#         self.export_path = export_path
#
#     @property
#     def done(self) -> bool:
#         return self.current_fold >= self.num_folds
#
#     def connect(self, fit_loop: FitLoop) -> None:
#         self.fit_loop = fit_loop
#
#     def reset(self) -> None:
#         """Nothing to reset in this loop."""
#
#     def on_run_start(self, *args: Any, **kwargs: Any) -> None:
#         """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
#         model."""
#
#         self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
#
#     def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
#         """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
#         print(f"STARTING FOLD {self.current_fold}")
#         # assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
#
#         self.trainer.datamodule.setup_fold_index(self.current_fold)
#
#     def advance(self, *args: Any, **kwargs: Any) -> None:
#         """Used to the run a fitting and testing on the current hold."""
#         self._reset_fitting()  # requires to reset the tracking stage.
#         self.fit_loop.run()
#
#         self._reset_testing()  # requires to reset the tracking stage.
#         # self.trainer.test_loop.run()
#         self.current_fold += 1  # increment fold tracking number.
#
#     def on_advance_end(self) -> None:
#         """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
#         self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
#         # restore the original weights + optimizers and schedulers.
#         self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
#         #  use distinct early stopping for current fold
#         # self.trainer.callbacks = [PrintCallback(),
#         #              EarlyStopping(monitor="val_loss_{}".format(self.current_fold),
#         #                            min_delta=0.0,
#         #                            check_on_train_epoch_end=True,
#         #                            patience=5,
#         #                            check_finite=True,
#         #                            # stopping_threshold=1e-4,
#         #                            mode='min'),
#         #              ]
#
#         self.trainer.checkpoint_connector = CheckpointConnector(trainer=self.trainer, resume_from_checkpoint=self.export_path)
#         self.trainer.state.fn = TrainerFn.FITTING
#         self.trainer.strategy.setup_optimizers(self.trainer)
#         self.replace(fit_loop=FitLoop)
#
#     def on_run_end(self) -> None:
#         """Used to compute the performance of the ensemble model on the test set."""
#         # checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
#         # scv_model = SpatialCVModel(type(self.trainer.lightning_module), checkpoint_paths)
#         # scv_model.trainer = self.trainer
#         # This requires to connect the new model and move it the right device.
#         # self.trainer.strategy.connect(scv_model)
#         # self.trainer.strategy.model_to_device()
#         # self.trainer.test_loop.run()
#
#     def on_save_checkpoint(self) -> Dict[str, int]:
#         return {"current_fold": self.current_fold}
#
#     def on_load_checkpoint(self, state_dict: Dict) -> None:
#         self.current_fold = state_dict["current_fold"]
#
#     def _reset_fitting(self) -> None:
#         self.trainer.reset_train_dataloader()
#         self.trainer.reset_val_dataloader()
#         self.trainer.state.fn = TrainerFn.FITTING
#         self.trainer.training = True
#
#     def _reset_testing(self) -> None:
#         self.trainer.reset_test_dataloader()
#         self.trainer.state.fn = TrainerFn.TESTING
#         self.trainer.testing = True
#
#     def __getattr__(self, key) -> Any:
#         # requires to be overridden as attributes of the wrapped loop are being accessed.
#         if key not in self.__dict__:
#             return getattr(self.fit_loop, key)
#         return self.__dict__[key]
