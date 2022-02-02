# -*- coding: utf-8 -*-
"""
1)
Given a date, merge all bands of a multi spectral image.
Then, spatially align images in ../1_Data_working_copy/MSI/ using the image taken on the 17.06.2020 as reference.
2)
Merge aligned files from 1) into one multi layer image with rescaling to coarsest resolution for all.

output folder for alignment: ../1_Data_working_copy/MSI/Aligned_Raster_Maps
"""

# Built-in/Generic Imports
import os
import shutil

# Libs
import pandas as pd
from arosics import COREG
from arosics import DESHIFTER
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


# path of root directory of multi-spectral images (MSI)
root = '../../1_Data_working_copy/MSI/'
# path to sub directory
sub_dir = '4_index/reflectance/'
# reference root directory name
reference_dir = 'pC_col_plant_drone_Multi_20200617'
# channels in reference image,
reference_files = ['Tempelberg_sequ_17062020_transparent_reflectance_green.tif',
                         'Tempelberg_sequ_17062020_transparent_reflectance_rededge.tif',
                         'Tempelberg_sequ_17062020_transparent_reflectance_nir.tif',
                         'Tempelberg_sequ_17062020_transparent_reflectance_red.tif']
# sort order of bands
bands = ['green', 'nir', 'red', 'rededge']


# assumption: given a date all channels within an image are aligned
# thus, we need to align the whole image rather than each channel separately
# merge reference image, if not exist
reference_merge_file = 'Tempelberg_sequ_17062020_transparent_reflectance_merged.tif'
reference_merge_file_abs = os.path.join(root, reference_dir, sub_dir, reference_merge_file)
if not os.path.isfile(reference_merge_file_abs):
    print('merging channels of {}'.format(reference_merge_file_abs))
    # Step 1.) Create a virtual raster (VRT) with option "separate=True" to stack the images as separate bands
    VRT = 'OutputImage.vrt'
    reference_files_abs = [os.path.join(root, reference_dir, sub_dir, x) for x in reference_files]
    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', separate=True, callback=gdal.TermProgress_nocb)
    gdal.BuildVRT(VRT, reference_files_abs, options=vrt_options)

    # Step 2.) Translate virtual raster (VRT) into GeoTiff:
    input_image = gdal.Open(VRT, 0)  # open the VRT in read-only mode
    gdal.Translate(reference_merge_file_abs, input_image, format='GTiff',
                   creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
                   callback=gdal.TermProgress_nocb)
    del input_image  # close the VRT
    del VRT
else:
    print('merged file found: {}'.format(reference_merge_file_abs))

alignment_path = os.path.join(root, 'Aligned_Raster_Maps')
if not os.path.exists(alignment_path):
    print('creating directory {}'.format(alignment_path))
    os.mkdir(alignment_path)

# merge target channels of images, compute alignment of target to reference image and align
for dir in os.listdir(root):
    dir = os.fsdecode(dir)
    if dir != reference_dir and dir != 'Aligned_Raster_Maps':
        # get all tif files in directory
        files = os.listdir(os.path.join(root, dir, sub_dir))
        files = [x for x in files if '.tif' in x]
        # we don't want to use blue and RGB, as these bands are only available for two drone flights
        files = [x for x in files if not 'blue' in x]
        files = [x for x in files if not 'RGB' in x]
        # also ignore DSM and NDVI at this stage and process later
        files = [x for x in files if not 'ndvi' in x]
        files = [x for x in files if not 'dsm' in x]

        target_file = os.fsdecode(files[0])
        target_merge_file = target_file.rsplit('_', 1)[0]+'_merged.tif'

        target_merge_file_abs = os.path.join(root, dir, sub_dir, target_merge_file)
        if not os.path.isfile(target_merge_file_abs):
            print('merging channels of {}'.format(target_merge_file_abs))
            # Step 1.) Create a virtual raster (VRT) with option "separate=True" to stack the images as separate bands
            VRT = 'OutputImage.vrt'

            if 'RGB' not in dir:
                # for MSI: create a dict and apply a particular order to bands
                file_bands=[file.split('_')[-1].split('.')[0] for file in files]
                index = dict(zip(file_bands,files))
                files = [index[band] for band in bands]

            files_abs = [os.path.join(root, dir, sub_dir, x) for x in files]
            vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', separate=True,
                                               callback=gdal.TermProgress_nocb)
            gdal.BuildVRT(VRT, files_abs, options=vrt_options)

            # Step 2.) Translate virtual raster (VRT) into GeoTiff:
            input_image = gdal.Open(VRT, 0)  # open the VRT in read-only mode
            gdal.Translate(target_merge_file_abs, input_image, format='GTiff',
                           creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
                           callback=gdal.TermProgress_nocb)
            del input_image  # close the VRT
            del VRT
        else:
            print('merged file found: {}'.format(target_merge_file_abs))

        # calculating alignment and perform shift for bands
        print('Aligning {target_file} to {reference_file}'.format(target_file=target_merge_file, reference_file=reference_merge_file))
        out_file = os.path.join(alignment_path,target_merge_file.split('.')[0]+'_aligned.tif')
        CR = COREG(reference_merge_file_abs, target_merge_file_abs, path_out=out_file, max_shift=11, ws=(256,256))
        CR.calculate_spatial_shifts()
        CR.correct_shifts()
        
        if 'RGB' not in dir:
            # apply same alignment to ndvi and dsm
            target_merge_file = target_file.rsplit('_', 3)[0] + '_index_ndvi.tif'
            out_file = os.path.join(alignment_path,target_file.rsplit('_', 3)[0] + '_index_ndvi_aligned.tif')
            print('Aligning {target_file}'.format(target_file=target_merge_file))
            print('writing to {out_file}'.format(out_file=out_file))
            target_merge_file_abs = os.path.join(root, dir, sub_dir, target_merge_file)
            DESHIFTER(target_merge_file_abs, CR.coreg_info, path_out=out_file,band2process=1).correct_shifts()

            target_merge_file = target_file.rsplit('_', 3)[0]+'_dsm.tif'
            out_file = os.path.join(alignment_path,target_file.rsplit('_', 3)[0]+'_dsm_aligned.tif')
            print('Aligning {target_file}'.format(target_file=target_merge_file))
            target_merge_file_abs = os.path.join(root, dir, sub_dir, target_merge_file)
            DESHIFTER(target_merge_file_abs, CR.coreg_info, path_out=out_file,band2process=1).correct_shifts()

        print(100 * '-')

# move merged reference file to alignment path
reference_path = os.path.join(root, reference_dir, sub_dir)
shutil.copyfile(os.path.join(reference_path, reference_merge_file), os.path.join(reference_path, reference_merge_file.split('.')[0]+'_aligned.tif'))
shutil.copyfile(os.path.join(reference_path, reference_merge_file), os.path.join(reference_path, reference_merge_file.split('.')[0]+'_aligned.tif'))
shutil.copyfile(os.path.join(reference_path, reference_merge_file), os.path.join(reference_path, reference_merge_file.split('.')[0]+'_aligned.tif'))
# files = os.listdir(reference_path)
# aligment_files = [file for file in files if 'aligned' in file]
# for aligment_file in aligment_files:
#     os.replace(os.path.join(reference_path, aligment_file), os.path.join(alignment_path, aligment_file))
