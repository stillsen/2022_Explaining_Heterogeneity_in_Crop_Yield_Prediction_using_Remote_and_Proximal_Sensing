# -*- coding: utf-8 -*-
"""
PyTorch Lightning Data Module Class
that works on files in an input folder, i.e. analysis folder f under
../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC .

For patch i:
1) load each corresponding filename_Patch_ID_XXX.tif file in that folder as VRT and upsample the resolution
   to 2977px x 2977px for no flower strip patches and to 2977px x 2480px with flower strips
2) combine all m VRTs with v_{i} bands each to a m*v_{i} tensor
3) spatially divide this tensor into k folds (for spatial cross validation)
4) generate samples in each fold of size 224px x 224px by sliding a window over the fold image with step size v;
   (no transform here)
5) OPTION: if multiple patches are in folder f, perform steps 1 to 4 for all patches and stack the folds
6) export each fold as pytorch tensor as .pt

output: Foldername.pt
"""

# Built-in/Generic Imports
import os
import shutil

# Libs
from osgeo import gdal
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Own modules

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


class PatchCROPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
