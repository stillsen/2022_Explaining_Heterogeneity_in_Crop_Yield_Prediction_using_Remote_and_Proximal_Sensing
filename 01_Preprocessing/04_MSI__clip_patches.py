# -*- coding: utf-8 -*-
"""
patchcrop whole landscape rasters in ../../1_Data_working_copy/MSI/Aligned_Raster_Maps differ with respect to
date and type (, i.e. 4band multi spectral image, DSM or NDVI)
-> clip from those patch sized rasters and save with _Patch_ID_XX_ tag
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

flowerstrip_patch_ids = ['12', '13', '19', '20', '21', '105', '110', '114', '115', '119']

# root = '../../1_Data_working_copy/MSI/Aligned_Raster_Maps'
# root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/1_Data_working_copy/MSI/pC_col_plant_drone_RGB3D_20200703/4_index/reflectance'
# root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/1_Data_working_copy/MSI/pC_col_plant-drone_RGB3D_20200806/4_index/reflectance'
# root = '/media/stillsen/Samsung_T5/pC2021/Tempelberg_AriaX_06072021_Haupt/3_dsm_ortho/2_mosaic/'
root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/1_Data_working_copy/MSI/Tempelberg_Haupt_Sequ_06072021/'

# for every every landscape file in folder....
landscape_files = [file for file in os.listdir(root) if file.endswith('tif')]
for landscape_file in landscape_files:
    # if not a clipped file already...
    if '_Patch_ID_' not in landscape_file:
        # open landscape file
        print('clipping {file}'.format(file=landscape_file))
        file_rel = os.path.join(root, landscape_file)
        file_handle = gdal.Open(file_rel)

        # loop over patch shape files to use them as spatial extent for clipping
        shape_files_rel = [os.path.join('../../1_Data_working_copy/Shapes/no_flower_strip/', shape) for shape in os.listdir('../../1_Data_working_copy/Shapes/no_flower_strip/') if shape.endswith('.shp')]
        shape_files_rel.extend([os.path.join('../../1_Data_working_copy/Shapes/with_flower_strip/', shape) for shape in os.listdir('../../1_Data_working_copy/Shapes/with_flower_strip/') if shape.endswith('.shp')])
        for shape_file_rel in shape_files_rel:
            # extract patch id
            patch_id = shape_file_rel.split('/')[-1].split('.')[0]
            outfile_rel = os.path.join(root, landscape_file.split('.')[0]+'_'+patch_id+'.tif')

            # shape_handle = gdal.Open(shape_file_rel)
            options = gdal.WarpOptions(cropToCutline=True, format="GTiff", cutlineDSName=shape_file_rel, dstNodata = 0)
            newfile = gdal.Warp(outfile_rel, file_handle, options=options)


# gdalwarp -s_srs EPSG:25833 -t_srs EPSG:25833 -of GTiff -cutline $f -crop_to_cutline -multi -dstnodata -9999 $rf "${PREFIX}_${filename}.tif"