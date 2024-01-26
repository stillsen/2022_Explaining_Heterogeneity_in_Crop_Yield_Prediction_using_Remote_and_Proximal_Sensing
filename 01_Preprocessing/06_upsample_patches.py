# -*- coding: utf-8 -*-
"""
for all geo tif files in folder under
../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC
upsample the resolution to 2977px x 2977px for no flower strip patches
and to 2977px x 2480px

The idea is to first upload images to the cluster and then to upsampled to reduce traffic significantly
"""

# Built-in/Generic Imports
import os
import shutil

# Libs
from osgeo import gdal

# Own modules

__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'

flowerstrip_patch_ids = ['Patch_ID_12', 'Patch_ID_13', 'Patch_ID_19', 'Patch_ID_20', 'Patch_ID_21', 'Patch_ID_105', 'Patch_ID_110', 'Patch_ID_114', 'Patch_ID_115', 'Patch_ID_119']
flowerstrip_patch_ids_other = ['PS412', 'PS413', 'PS419', 'PS420', 'PS421', 'PS4105', 'PS4110', 'PS4114', 'PS4115', 'PS4119']

# root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC'
root = '/beegfs/stiller/PatchCROP_all/Data'

# for every every experiment in folder....
for dir in os.listdir(root):
    # upsample resolution of each raster file contained in dir, to the appropriate resolution (s.o.)
    files = os.listdir(os.path.join(root, dir))
    if 'Patch_ID_95' in dir:
        for file in files:
            # if file == 'Tempelberg_soda3D_03072020_transparent_mosaic_group1_Patch_ID_73.tif' or file == 'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif' or file == 'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif' or file == 'pC_col_2020_plant_PS473_SOats_smc_Krig.tif':
                # not an upsampled file ...
            if file.endswith('tif'):
                # if flower strip patch
                if any([fpids in file for fpids in flowerstrip_patch_ids]) or any([fpids in file for fpids in flowerstrip_patch_ids_other]):
                    (x, y) = (2977, 2480)
                else:
                    (x, y) = (2977, 2977)
                file_rel = os.path.join(root, dir, file)
                # outfile_rel = os.path.join(root, dir, file.split('.')[0]+'_upsampled.tif')
                outfile_rel = os.path.join(root, dir, file.split('.')[0] + '.tif')
                print('upsampling {file} to ({x},{y})'.format(file=file, x=x, y=y))
                file_handle = gdal.Open(file_rel)
                options = gdal.WarpOptions(format="GTiff", width=x, height=y, resampleAlg='cubicspline')
                newfile = gdal.Warp(outfile_rel, file_handle, options=options)


