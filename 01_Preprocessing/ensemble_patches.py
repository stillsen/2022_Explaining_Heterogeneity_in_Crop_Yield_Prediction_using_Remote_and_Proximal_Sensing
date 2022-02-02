# -*- coding: utf-8 -*-
"""
copy features and labels for each patch to ../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/Patch_ID;
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
target_root_path = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
label_path = '../../2_Data_preprocessed/58x_Kriging_Interpolated_Yield_Maps/'
# soil_texture_path = ''
# kompost_path = ''

# Lupine patches
patches = ['12', '13', '19', '20', '21', '39', '40', '49', '50', '51', '58', '59', '60', '65', '66', '68', '73', '74', '76', '81', '89', '90', '95', '96', '102', '105', '110', '114', '115', '119']
patches_id = ['Patch_ID_'+pid for pid in patches]
## Features
# copying remote sensing images
for file in os.listdir(MSI_path):
    if file.endswith('.tif'):
        # if file belongs to patch, copy
        if any(patch in file for patch in patches_id):
            # extract Patch_ID_76 out of Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_76.tif
            patch_id = '{}_{}_{}'.format(*file.split('.')[0].rsplit('_',3)[-3:])
            target_path_rel = os.path.join(target_root_path, patch_id)
            if not os.path.isdir(target_path_rel):
                print('creating directory {}'.format(target_path_rel))
                os.mkdir(target_path_rel)
            print('copying {file}'.format(file=file))
            if not os.path.isfile(os.path.join(target_path_rel, file)):
                shutil.copyfile(os.path.join(MSI_path, file), os.path.join(target_path_rel, file))

## Labels
# copying low resolution yield maps
for file in os.listdir(label_path):
    if file.endswith('.tif'):
        # skip if any of the listed subfolders
        if any([dir in file for dir in ['nfs', 'wfs']]):
            continue
        # if file belongs to patch, copy
        if any(patch in file for patch in patches):
            patch_id = 'Patch_ID_'+file.split('_')[4][3:]
            target_path_rel = os.path.join(target_root_path, patch_id)
            print('copying {file}'.format(file=file))
            if not os.path.isfile(os.path.join(target_path_rel, file)):
                shutil.copyfile(os.path.join(label_path, file), os.path.join(target_path_rel, file))

