# -*- coding: utf-8 -*-
"""
Data module::




works on files in an input folder
self.data_dir

, i.e. analysis folder f under
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
import warnings
import random
import math

# Libs
import pandas as pd
from osgeo import gdal
import numpy as np
from typing import Optional

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ConcatDataset, TensorDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from matplotlib import pyplot as plt

# Own modules
from TransformTensorDataset import TransformTensorDataset

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '1.0'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


class PatchCROPDataModule:
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None
    test_fold: Optional[Dataset] = None

    def __init__(self,
                 input_files: dict,
                 patch_id,
                 this_output_dir,
                 seed,
                 data_dir: str = './',
                 batch_size: int = 20,
                 stride: int = 10,
                 workers: int = 0,
                 input_features: str = 'RGB',
                 augmented: bool = False,
                 validation_strategy: str = 'SCV',
                 fake_labels: bool = False,
                 ):
        super().__init__()
        self.flowerstrip_patch_ids = [12, 13, 19, 20, 21, 105, 110, 114, 115, 119]
        self.patch_ids = ['12', '13', '19', '20', '21', '39', '40', '49', '50', '51', '58', '59', '60', '65', '66', '68', '73', '74', '76', '81', '89', '90', '95', '96', '102', '105', '110', '114',
                          '115', '119']

        self.data_dir = data_dir
        # name of the label file
        self.input_files = input_files
        self.batch_size = batch_size
        self.stride = stride
        self.workers = workers
        self.augmented = augmented
        self.patch_id = patch_id
        self.input_features = input_features  # options: a) RGB - red, green, blue; b) GRN - green, red edge, near infrared; c) RGBRN- red, green, blue, red edge, near infrafred
        self.validation_strategy = validation_strategy
        self.fake_labels = fake_labels
        self.training_response_standardizer = None
        self.fold = None
        self.this_output_dir = this_output_dir
        self.seed = seed
        self.val_idxs = None
        self.train_idxs = None

        # Datesets
        self.data_set = None
        self.setup_done = False

        if self.input_features == 'RGB':
            self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        elif self.input_features == 'GRN':
            self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            warnings.warn('Channel normalization for GRN most likely different from RGB')
        elif self.input_features == 'RGBRN':
            self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406, 0, 0],
                                         std=[0.229, 0.224, 0.225, 1, 1])
            raise NotImplemented("Channel normalization not implemented for RGBRN")

    def prepare_data(self, num_samples: int = None) -> None:  # would actually make more sense to put it to setup, but there it is called as several processes
        # load the data
        # normalize
        # split
        # oversample
        # load feature label combinations

        if self.fake_labels:
            kfold_dir = os.path.join(self.data_dir, 'kfold_set_fakelabels_s{}'.format(self.stride))
        else:
            kfold_dir = os.path.join(self.data_dir, 'kfold_set_origlabels_s{}'.format(self.stride))

        if not self.setup_done:
            # check if already build and load
            if os.path.isdir(kfold_dir):
                # load raw remote sensing images
                print('loading data')
                if self.input_features == "RGB":
                    file_name = 'Patch_ID_' + str(self.patch_id) + '.pt'
                elif self.input_features == "GRN":
                    file_name = 'Patch_ID_' + str(self.patch_id) + '_grn.pt'
                elif self.input_features == "RGBRN":
                    file_name = 'Patch_ID_' + str(self.patch_id) + '_RGBRN.pt'

                # f = torch.load(os.path.join(kfold_dir,file), map_location=torch.device('cuda'))
                f = torch.load(os.path.join(kfold_dir, file_name))
                self.data_set = f
            else:
                # otherwise build oversampled k-fold set
                print('generating data')
                for label, features in self.input_files.items():
                    label_matrix = torch.tensor([])
                    feature_matrix = torch.tensor([])
                    # load label
                    label_matrix = torch.tensor(gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray(), dtype=torch.float)
                    # label_matrix = gdal.Open(os.path.join(self.data_dir, label), gdal.GA_ReadOnly).ReadAsArray()
                    # load and concat features
                    for idx, feature in enumerate(features):
                        # if RGB drop alpha channel, else all
                        if 'soda' in feature or 'Soda' in feature:  # RGB
                            print(feature)
                            f = torch.tensor(gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray()[:3, :, :], dtype=torch.float)  # only first 3 channels -> not alpha channel
                            # f = gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray()[:3, :,:]  # only first 3 channels -> not alpha channel
                        else:  # multichannel
                            f = torch.tensor(gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray(), dtype=torch.float)
                            # f = gdal.Open(os.path.join(self.data_dir, feature), gdal.GA_ReadOnly).ReadAsArray()
                        # concat feature tensor, e.g. RGB tensor and NDVI tensor
                        if f.dim() == 2:
                            feature_matrix = torch.cat((feature_matrix, f.unsqueeze(0)), 0)
                        else:
                            feature_matrix = torch.cat((feature_matrix, f), 0)
                    # sliding window  -> n samples of 224x224 images
                    # creates self.data_set
                    self.generate_sliding_window_augmented_ds(label_matrix, feature_matrix)

                # save data_set
                os.mkdir(kfold_dir)
                file_name = kfold_dir + '/' + self.data_dir.split('/')[-1] + '.pt'
                torch.save(self.data_set, file_name)
            self.setup_done = True

    def generate_sliding_window_augmented_ds(self, label_matrix, feature_matrix, lower_bound: int = 327, kernel_size: int = 224) -> None:
        '''
        :param label_matrix: kriging interpolated yield maps
        :param feature_matrix: stack of feature maps, such as rgb remote sensing, multi-channel remote sensing or dsm or ndvi
        both are required to have same dimension, but have different size according to whether the patch is a flower strip
        or not
        :return: None
        generate sliding window samples over whole patch and save as self.data_set
        '''
        # length= 2322/1935

        upper_bound_x = feature_matrix.shape[2] - lower_bound
        upper_bound_y = feature_matrix.shape[1] - lower_bound

        # clip to tightest inner rectangle
        feature_arr = feature_matrix[:, lower_bound:upper_bound_y, lower_bound:upper_bound_x]

        # scale to 0...1
        if self.input_features == 'RGB':
            # feature_arr = feature_arr / 255 # --> deprecated, normalization is now done in setup_fold()->transformer
            pass
        elif self.input_features == 'GRN':
            c1_min = feature_arr[0, :, :].min()
            c2_min = feature_arr[1, :, :].min()
            c3_min = feature_arr[2, :, :].min()
            c1_max = feature_arr[0, :, :].max()
            c2_max = feature_arr[1, :, :].max()
            c3_max = feature_arr[2, :, :].max()

            feature_arr[0, :, :] = ((feature_arr[0, :,
                                     :] - c1_min) / c1_max) * 255  # normalization to [0..1] *255 to first get it to RGB range, second use normalization is now done in setup_fold()->transformer
            feature_arr[1, :, :] = ((feature_arr[1, :, :] - c2_min) / c2_max) * 255
            feature_arr[2, :, :] = ((feature_arr[2, :, :] - c3_min) / c3_max) * 255
        elif self.input_features == 'RGBRN':
            c4_min = feature_arr[3, :, :].min()
            c5_min = feature_arr[4, :, :].min()
            c4_max = feature_arr[3, :, :].max()
            c5_max = feature_arr[4, :, :].max()

            feature_arr[3, :, :] = ((feature_arr[3, :,
                                     :] - c4_min) / c4_max) * 255  # normalization to [0..1] *255 to first get it to RGB range, second use normalization is now done in setup_fold()->transformer
            feature_arr[4, :, :] = ((feature_arr[4, :, :] - c5_min) / c5_max) * 255

        if self.fake_labels:
            # three channel pixel-wise average -> but this is still in range [0..255]
            label_arr = (feature_arr[0, :, :] + feature_arr[1, :, :] + feature_arr[2, :, :]) / 3
            # normalize to range [0..255]
            label_arr = label_arr / 255
        else:
            label_arr = label_matrix[lower_bound:upper_bound_y, lower_bound:upper_bound_x]  # label matrix un-normalized
        # set sizes
        self.data_set_row_extend = feature_arr.shape[2]
        self.data_set_column_extend = feature_arr.shape[1]

        # calculating start positions of kernels
        # x and y are need to able to take different values, as patches with flowerstrips are smaller
        x_size = feature_arr.shape[2]
        y_size = feature_arr.shape[1]
        possible_shifts_x = int((x_size - kernel_size) / self.stride)
        x_starts = np.array(list(range(possible_shifts_x))) * self.stride
        possible_shifts_y = int((y_size - kernel_size) / self.stride)
        y_starts = np.array(list(range(possible_shifts_y))) * self.stride

        # loop over start postitions and save kernel as separate image
        feature_kernel_tensor = None
        label_kernel_tensor = None
        for y in y_starts:
            print('y: {}'.format(y))
            if y == y_starts[int(len(y_starts) / 4)]:
                print("........... 25%")
            if y == y_starts[int(len(y_starts) / 2)]:
                print("........... 50%")
            if y == y_starts[int(len(y_starts) * 3 / 4)]:
                print("........... 75%")
            for x in x_starts:
                # shift kernel over image and extract kernel part
                # only take RGB value
                feature_kernel_img = feature_arr[:, y:y + kernel_size, x:x + kernel_size]
                label_kernel_img = label_arr[y:y + kernel_size, x:x + kernel_size]
                if x == 0 and y == 0:
                    feature_kernel_tensor = feature_kernel_img.unsqueeze(0)
                    label_kernel_tensor = label_kernel_img.mean().unsqueeze(0)
                else:
                    feature_kernel_tensor = torch.cat((feature_kernel_tensor, feature_kernel_img.unsqueeze(0)), 0)
                    label_kernel_tensor = torch.cat((label_kernel_tensor, label_kernel_img.mean().unsqueeze(0)), 0)
        self.data_set = (feature_kernel_tensor, label_kernel_tensor)

    def setup_fold(self, fold: int = 0, training_response_standardization: bool = True, test_transforms=None) -> None:
        '''
        Set up validation and train data set using self.data_set and apply normalization for both and augmentations for the train set if
        self.augmented. Overlap samples between train and test set are discarded.

        :return: None
        '''

        print('\tSetting up fold specific datasets...')

        self.fold = fold
        val_set = None
        train_set = None
        test_set = None

        if self.load_subset_indices(k=fold):
            val_idxs = self.val_idxs
            train_idxs= self.train_idxs
        else:
            # Splitting into datasets according to validation strategy
            # Part 1) Compute subset indexes
            if self.validation_strategy == 'SCV' or self.validation_strategy == 'SCV_no_test':  # spatial cross validation with or without test set
                print('splitting for SCV')
                # ranges for x and y
                # quadrants:
                # 0 1
                # 2 3
                # test val train   | fold
                #  0    1   {2,3}  |  0
                #  1    3   {0,2}  |  1
                #  3    2   {0,1}  |  2
                #  2    0   {1,3}  |  3

                # sliding window size
                window_size = 224
                stride = self.stride
                x_size = y_size = 2322

                # number of sliding window samples in any given row
                samples_per_row = samples_per_col = math.floor((x_size - window_size) / stride)

                x_def = y_def = np.arange(samples_per_row)
                # center lines to split the patch into folds at
                center_x = center_y = x_size / 2
                # last row/col index such that samples do not overlap between folds in quadrant 0/2
                buffer_to_x = buffer_to_y = math.floor((math.floor(center_x) - window_size) / stride)
                # first row/col index such that samples do not overlap between folds in quadrant 1/3
                buffer_from_x = buffer_from_y = math.ceil(math.floor(center_x) / stride)
                # last row/col index such that samples CAN OVERLAP between folds in quadrant 0/2
                overlap_to_x = overlap_to_y = math.floor((math.floor(center_x)) / stride)
                # first row/col index such that samples CAN OVERLAP between folds in quadrant 1/3
                overlap_from_x = overlap_from_y = math.ceil(math.floor(center_x) / stride)
                if self.patch_id in self.flowerstrip_patch_ids:
                    y_size = 1935
                    center_y = y_size / 2
                    samples_per_col = math.floor((y_size - window_size) / stride)
                    y_def = np.arange(samples_per_col)
                    buffer_to_y = math.floor((math.floor(center_y) - window_size) / stride)
                    buffer_from_y = math.ceil(math.floor(center_y) / stride)
                    overlap_to_y = math.floor((math.floor(center_y)) / stride)
                    # buffer_from_y = math.ceil(math.floor(center_y) / stride)
                    overlap_from_y = math.ceil(math.floor(center_y) / stride)

                debug_test_map = [np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  np.zeros((1, y_size, x_size)),
                                  ]
                debug_val_map = [np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 np.zeros((1, y_size, x_size)),
                                 ]
                debug_train_map = [np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   np.zeros((1, y_size, x_size)),
                                   ]
                # no overlap quadrants
                quadrant_0 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(buffer_to_y)]
                quadrant_1 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(buffer_to_y)]
                quadrant_2 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(buffer_from_y, samples_per_col)]
                quadrant_3 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(buffer_from_y, samples_per_col)]

                # overlap quadrants
                overlap_quadrant_0 = [(x, y) for x in np.arange(overlap_to_x) for y in np.arange(overlap_to_y)]
                overlap_quadrant_1 = [(x, y) for x in np.arange(overlap_from_x, samples_per_row) for y in np.arange(overlap_to_y)]
                overlap_quadrant_2 = [(x, y) for x in np.arange(overlap_to_x) for y in np.arange(overlap_from_y, samples_per_col)]
                overlap_quadrant_3 = [(x, y) for x in np.arange(overlap_from_x, samples_per_row) for y in np.arange(overlap_from_y, samples_per_col)]

                half_23 = [(x, y) for x in np.arange(samples_per_row) for y in np.arange(buffer_from_y, samples_per_col)]
                half_02 = [(x, y) for x in np.arange(buffer_to_x) for y in np.arange(samples_per_col)]
                half_01 = [(x, y) for x in np.arange(samples_per_row) for y in np.arange(buffer_to_y)]
                half_13 = [(x, y) for x in np.arange(buffer_from_x, samples_per_row) for y in np.arange(samples_per_col)]

                if self.validation_strategy == 'SCV':  # spatial cross validation with test set
                    test_range = [quadrant_0,  # quadrant 0
                                  quadrant_1,  # quadrant 1
                                  quadrant_3,  # quadrant 3
                                  quadrant_2,  # quadrant 2
                                  ]
                    val_range = [quadrant_1,  # quadrant 1
                                 quadrant_3,  # quadrant 3
                                 quadrant_2,  # quadrant 2
                                 quadrant_0,  # quadrant 0
                                 ]
                    train_range = [half_23,
                                   half_02,
                                   half_01,
                                   half_13,
                                   ]

                    test_idxs = []
                    val_idxs = []
                    train_idxs = []

                    for y in y_def:
                        for x in x_def:
                            idx = (x + y * samples_per_row)
                            if (x, y) in test_range[fold]:
                                test_idxs.append(idx)
                                debug_test_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in val_range[fold]:
                                val_idxs.append(idx)
                                debug_val_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in train_range[fold]:
                                train_idxs.append(idx)
                                debug_train_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                    self.save_subset_indices(k=fold, train_indices=train_idxs, val_indices=val_idxs)
                elif self.validation_strategy == 'SCV_no_test':  # spatial cross validation no test set
                    threequarter_230 = list(set(half_02 + half_23))
                    threequarter_021 = list(set(half_02 + half_01))
                    threequarter_013 = list(set(half_01 + half_13))
                    threequarter_132 = list(set(half_13 + half_23))

                    val_range = [quadrant_1,  # quadrant 1
                                 quadrant_3,  # quadrant 3
                                 quadrant_2,  # quadrant 2
                                 quadrant_0,  # quadrant 0
                                 ]
                    train_range = [threequarter_230,
                                   threequarter_021,
                                   threequarter_013,
                                   threequarter_132,
                                   ]

                    test_idxs = []
                    val_idxs = []
                    train_idxs = []

                    for y in y_def:
                        for x in x_def:
                            idx = (x + y * samples_per_row)
                            if (x, y) in val_range[fold]:
                                val_idxs.append(idx)
                                debug_val_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                            elif (x, y) in train_range[fold]:
                                train_idxs.append(idx)
                                debug_train_map[fold][0, y * stride:y * stride + window_size, x * stride:x * stride + window_size] += 1
                # debug images to verify correct splitting and amount sample
                # create folders
                print('creating debug fold images')
                dir = os.path.join(self.data_dir, 'Test_Figs')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                # val
                m = np.max(debug_val_map[fold])
                debug_val_map[fold] = debug_val_map[fold]  # /m
                fig, ax = plt.subplots()
                c = plt.imshow(debug_val_map[fold][0, :, :], cmap='jet')
                fig.colorbar(c, ax=ax)
                ax.xaxis.tick_top()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, 'val_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                # train
                m = np.max(debug_train_map[fold])
                debug_train_map[fold] = debug_train_map[fold]  # / m
                fig, ax = plt.subplots()
                c = plt.imshow(debug_train_map[fold][0, :, :], cmap='jet')
                fig.colorbar(c, ax=ax)
                ax.xaxis.tick_top()
                plt.tight_layout()
                plt.savefig(os.path.join(dir, 'train_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                # test
                if self.validation_strategy == 'SCV':
                    m = np.max(debug_test_map[fold])
                    debug_test_map[fold] = debug_test_map[fold]  # / m
                    fig, ax = plt.subplots()
                    c = plt.imshow(debug_test_map[fold][0, :, :], cmap='jet')
                    fig.colorbar(c, ax=ax)
                    ax.xaxis.tick_top()
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, 'test_folds_{}_{}_{}.png'.format(fold, stride, self.validation_strategy)))
                self.save_subset_indices(k=fold, train_indices=train_idxs, val_indices=val_idxs)
            elif self.validation_strategy == 'SHOV':  # spatial hold out validation
                # train_idxs = np.arange(0,61)
                # val_idxs = np.arange(61, 81)
                # train_idxs = np.arange(0, 234)
                # val_idxs = np.arange(252, 324)
                # train_idxs = np.arange(0, 989)
                # val_idxs = np.arange(1027, 1369)
                # stride 30
                train_idxs = np.arange(0, 3294)
                val_idxs = np.arange(3847, 4761)
                self.save_subset_indices(k=fold, train_indices=train_idxs, val_indices=val_idxs)
            elif self.validation_strategy == 'RCV':  # random CV
                kf = KFold(n_splits=4, random_state=self.seed, shuffle=True)
                idxs = np.arange(len(self.data_set[0]))
                k = 0
                for train_idxs , val_idxs in kf.split(X=idxs):
                    self.save_subset_indices(k=k, train_indices=train_idxs, val_indices=val_idxs)
                    k += 1
                # non_spatial_cv_split_idxs = train_test_split(idxs, random_state=self.seed)
                # train_idxs = non_spatial_cv_split_idxs[0]
                # val_idxs = non_spatial_cv_split_idxs[1]


        # subset the dataset into train_set, test_set and val_set
        if training_response_standardization:
            # standardize response variables according to mean and std of train set
            self.training_response_standardizer = {'mean': torch.mean(self.data_set[1][train_idxs]),
                                                   'std': torch.std(self.data_set[1][train_idxs]),
                                                   }
            train_set = (self.data_set[0][train_idxs, :, :, :],
                         (self.data_set[1][train_idxs] - self.training_response_standardizer['mean']) / self.training_response_standardizer['std'])
            val_set = (self.data_set[0][val_idxs, :, :, :],
                       (self.data_set[1][val_idxs] - self.training_response_standardizer['mean']) / self.training_response_standardizer['std'])
            if self.validation_strategy == 'SCV':
                test_set = (self.data_set[0][test_idxs, :, :, :],
                            (self.data_set[1][test_idxs] - self.training_response_standardizer['mean']) / self.training_response_standardizer['std'])
        else:
            # train_set = (self.data_set[0][train_idxs, :, :, :].repeat(duplicate_trainset_ntimes, 1, 1, 1), self.data_set[1][train_idxs].repeat(duplicate_trainset_ntimes))
            train_set = (self.data_set[0][train_idxs, :, :, :], self.data_set[1][train_idxs])
            val_set = (self.data_set[0][val_idxs, :, :, :], self.data_set[1][val_idxs])
            if self.validation_strategy == 'SCV':
                test_set = (self.data_set[0][test_idxs, :, :, :], self.data_set[1][test_idxs])

        if test_transforms is None:
            testset_transformer = A.Compose([  # assumption by albumentations: image is in HWC
                self.normalize,
                ToTensorV2(),  # convert to CHW
            ])
        else:
            testset_transformer = test_transforms

        # type cast to numpy again for compatibility with albumentations\
        # val_set = (val_set[0].moveaxis(1,3).numpy(), val_set[1].numpy())
        # test_set = (test_set[0].moveaxis(1,3).numpy(), test_set[1].numpy())
        val_set = (val_set[0].movedim(1, 3).numpy(), val_set[1].numpy())

        self.val_fold = TransformTensorDataset(val_set, transform=testset_transformer)
        if self.validation_strategy == 'SCV':
            test_set = (test_set[0].movedim(1, 3).numpy(), test_set[1].numpy())
            self.test_fold = TransformTensorDataset(test_set, transform=testset_transformer)

        # only train set is augmented if enabled
        if self.augmented:
            trainset_transformer = A.Compose([  # transforms.ToPILImage(),
                self.normalize,
                # A.Blur(),
                # A.CLAHE(),
                A.RandomBrightnessContrast(),
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Transpose(),
                ToTensorV2(),
            ])
        else:
            trainset_transformer = A.Compose([  # transforms.ToPILImage(),
                self.normalize,
                ToTensorV2(),
            ])
        # type cast to numpy again for compatibility with albumentations\
        train_set = (train_set[0].movedim(1, 3).numpy(), train_set[1].numpy())
        self.train_fold = TransformTensorDataset(train_set, transform=trainset_transformer)
        print('\tSet up done...')

    def load_subset_indices(self, k):
        val_filename = os.path.join(self.this_output_dir, 'val_df_' + str(k) + self.validation_strategy + '.csv')
        train_filename = os.path.join(self.this_output_dir, 'train_df_' + str(k) + self.validation_strategy + '.csv')
        if os.path.exists(val_filename):
            self.val_idxs = pd.read_csv(val_filename, usecols=['val_indices'])['val_indices'].values
            self.train_idxs = pd.read_csv(train_filename, usecols=['train_indices'])['train_indices'].values
            return True
        return False

    def save_subset_indices(self, k, train_indices, val_indices):
        train_df = pd.DataFrame({'train_indices': train_indices})
        val_df = pd.DataFrame({'val_indices': val_indices, })
        train_df.to_csv(os.path.join(self.this_output_dir, 'train_df_' + str(k) + self.validation_strategy + '.csv'), encoding='utf-8')
        val_df.to_csv(os.path.join(self.this_output_dir, 'val_df_' + str(k) + self.validation_strategy + '.csv'), encoding='utf-8')

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        print('loading training set with {} samples'.format(len(self.train_fold)))
        return DataLoader(self.train_fold, shuffle=True, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self) -> DataLoader:
        print('loading validation set with {} samples'.format(len(self.val_fold)))
        return DataLoader(self.val_fold, num_workers=self.workers, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        print('loading validation set with {} samples'.format(len(self.test_fold)))
        return DataLoader(self.test_fold, num_workers=self.workers, batch_size=self.batch_size)

    def all_dataloader(self) -> DataLoader:
        print('loading whole patch data set with {} samples'.format(len(self.data_set)))
        transformer = A.Compose([self.normalize,
                                 ToTensorV2(),
                                 ])
        data_set = (self.data_set[0].movedim(1, 3).numpy(), self.data_set[1].numpy())
        whole_ds = TransformTensorDataset(data_set, transform=transformer)

        return DataLoader(whole_ds, num_workers=self.workers, batch_size=self.batch_size)

    def create_debug_samples(self, n=20):
        '''
        Save n samples from each data set to folder, for purposes of testing if augmentations are correctly applied
        :param n: number of samples saved to self.data_dir/Test_Figs for train/val/test set
        :return:
        '''
        # create folders
        dir = os.path.join(self.data_dir, 'Test_Figs')
        if not os.path.exists(dir):
            os.mkdir(dir)

        # load dataset loaders with batch_size = 1
        orig_bs = self.batch_size
        self.batch_size = 1
        dataloaders = {'train': self.train_dataloader(),
                       'val': self.val_dataloader(),
                       # 'test': self.test_dataloader()
                       }

        for phase in ['train', 'val']:
            print('\tSaving Debug Figures for {}'.format(phase))
            i = 0
            c1 = None
            c2 = None
            c3 = None
            if self.input_features == 'RGBRN':
                c4 = None
                c5 = None
            response = None
            for inputs, labels in dataloaders[phase]:
                # save sample figures
                if i < n:
                    plt.imshow(inputs.squeeze().movedim(0, 2).numpy())
                    plt.savefig(os.path.join(dir, 'fold{}_{}_{}_{}.png'.format(self.fold, phase, i, str(labels.squeeze().item())[:5])))
                    i += 1
        self.batch_size = orig_bs

    def __post_init__(cls):
        super().__init__()

