# -*- coding: utf-8 -*-
"""
copy features and labels for Lupine to ../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Lupine;
if file does not already exist, else skip
"""

# Built-in/Generic Imports
import os
import shutil

# Libs

# Own modules

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

MSI_path = '../../1_Data_working_copy/MSI/Aligned_Raster_Maps/'
target_path = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Lupine/'
label_path = '../../2_Data_preprocessed/58x_Kriging_Interpolated_Yield_Maps/'
# soil_texture_path = ''
# kompost_path = ''

# Lupine patches
patches = ['59', '89', '119']

## Features
# copying remote sensing images
for file in os.listdir(MSI_path):
    if file.endswith('.tif'):
        # if file belongs to patch, copy
        if any(patch in file for patch in patches):
            print('copying {file}'.format(file=file))
            if not os.path.isfile(os.path.join(target_path, file)):
                shutil.copyfile(os.path.join(MSI_path, file), os.path.join(target_path, file))

## Labels
# copying low resolution yield maps
for file in os.listdir(label_path):
    if file.endswith('.tif'):
        # skip if any of the listed subfolders
        if any([dir in file for dir in ['nfs', 'wfs']]):
            continue
        # if file belongs to patch, copy
        if any(patch in file for patch in patches):
            print('copying {file}'.format(file=file))
            if not os.path.isfile(os.path.join(target_path, file)):
                shutil.copyfile(os.path.join(label_path, file), os.path.join(target_path, file))

