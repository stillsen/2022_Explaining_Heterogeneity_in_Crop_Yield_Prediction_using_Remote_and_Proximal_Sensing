
import os


output_dirs = dict()
data_dirs = dict()
input_files = dict()
input_files_rgb = dict()

data_root = '/beegfs/stiller/PatchCROP_all/Data/'
# data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
output_root = '/beegfs/stiller/PatchCROP_all/Output/'
# output_root = '../../Output/'

# ## Patch 12
# output_dirs[12] = os.path.join(output_root, 'P_12')
# data_dirs[12] = os.path.join(data_root, 'Patch_ID_12')
# # input_files[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_12.tif',
# #                                                                 'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_12.tif',
# #                                                                 'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_12.tif',
# #                                                                 'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']                   }
# input_files_rgb[12] = {'pC_col_2020_plant_PS412_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_12.tif']}
#
# ## Patch 19
# output_dirs[19] = os.path.join(output_root, 'P_19')
# data_dirs[19] = os.path.join(data_root, 'Patch_ID_19')
# input_files_rgb[19] = {'pC_col_2020_plant_PS419_Soy_smc_Krig.tif': [
#     'Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_19.tif']}
# input_files_rgb[19] = {'pC_col_2020_plant_PS419_Soy_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_19.tif',
#                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_19.tif',
#                                                                       ]}

# ## Patch 39
# output_dirs[39] = os.path.join(output_root, 'P_39')
# data_dirs[39] = os.path.join(data_root, 'Patch_ID_39')
# # input_files[39] = {
# #     'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_39.tif',
# #                                                    'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_39.tif',
# #                                                    'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_39.tif',
# #                                                    'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}
# input_files_rgb[39] = {'pC_col_2020_plant_PS439_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_39.tif']}
#
# ## Patch 50
# output_dirs[50] = os.path.join(output_root, 'P_50')
# data_dirs[50] = os.path.join(data_root, 'Patch_ID_50')
# input_files_rgb[50] = {'pC_col_2020_plant_PS450_Maiz_smc_Krig.tif': [
#     'Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_50.tif']}
# # input_files_rgb[50] = {'pC_col_2020_plant_PS450_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_50.tif',
# #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_50.tif',
# #                                                                       ]}
#
# ## Patch 58
# output_dirs[58] = os.path.join(output_root, 'P_58')
# data_dirs[58] = os.path.join(data_root, 'Patch_ID_58')
# input_files_rgb[58] = {'pC_col_2020_plant_PS458_Soy_smc_Krig.tif': [
#     'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_58.tif']}
# # input_files_rgb[50] = {'pC_col_2020_plant_PS450_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_50.tif',
# #                                                                       'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_50.tif',
# #                                                                       ]}
## Patch 19
output_dirs[19] = os.path.join(output_root, 'P_19')
data_dirs[19] = os.path.join(data_root, 'Patch_ID_19')
input_files_rgb[19] = {'pC_col_2020_plant_PS419_Soy_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_19.tif']}

## Patch 65
output_dirs[65] = os.path.join(output_root, 'P_65')
data_dirs[65] = os.path.join(data_root, 'Patch_ID_65')
input_files_rgb[65] = {'pC_col_2020_plant_PS465_Soy_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_65.tif']}

## Patch 68
output_dirs[68] = os.path.join(output_root, 'P_68')
data_dirs[68] = os.path.join(data_root, 'Patch_ID_68')
# input_files[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
#                                                                   'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
#                                                                   'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
#                                                                   'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
#                    }
input_files_rgb[68] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_68.tif']}

## Patch 68
output_dirs['68_grn'] = os.path.join(output_root, 'P_68')
data_dirs['68_grn'] = os.path.join(data_root, 'Patch_ID_68_grn')
input_files_rgb['68_grn'] = {'pC_col_2020_plant_PS468_Maiz_smc_Krig.tif': ['Tempelberg_sequ_16072020_transparent_reflectance_merged_green_nir_re_Patch_ID_68.tif']}

# ## Patch 73
# output_dirs[73] = os.path.join(output_root, 'P_73')
# data_dirs[73] = os.path.join(data_root, 'Patch_ID_73')
# # input_files[73] = {
# #     'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_73.tif',
# #                                                    'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_73.tif',
# #                                                    'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_73.tif',
# #                                                    'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']
# #     }
# input_files_rgb[73] = {'pC_col_2020_plant_PS473_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_73.tif']}

## Patch 74
output_dirs[74] = os.path.join(output_root, 'P_74')
data_dirs[74] = os.path.join(data_root, 'Patch_ID_74')
input_files_rgb[74] = {'pC_col_2020_plant_PS474_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_74.tif']}

## Patch 76
output_dirs[76] = os.path.join(output_root, 'P_76')
data_dirs[76] = os.path.join(data_root, 'Patch_ID_76')
input_files_rgb[76] = {'pC_col_2020_plant_PS476_Sun_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_76.tif']}

#
# ## Patch 81
# output_dirs[81] = os.path.join(output_root, 'P_81')
# data_dirs[81] = os.path.join(data_root, 'Patch_ID_81')
# input_files_rgb[81] = {'pC_col_2020_plant_PS481_SOats_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_81.tif']}
#
# ## Patch 90
# output_dirs['90_grn'] = os.path.join(output_root, 'P_90_falsecolor')
# data_dirs['90_grn'] = os.path.join(data_root, 'Patch_ID_90_grn')
# input_files_rgb['90_grn'] = {'pC_col_2020_plant_PS490_Maiz_smc_Krig.tif': ['Tempelberg_sequ_16072020_transparent_reflectance_merged_green_nir_re_Patch_ID_90.tif']}

## Patch 90
output_dirs[90] = os.path.join(output_root, 'P_90')
data_dirs[90] = os.path.join(data_root, 'Patch_ID_90')
input_files_rgb[90] = {'pC_col_2020_plant_PS490_Maiz_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_90.tif']}

## Patch 95
output_dirs[95] = os.path.join(output_root, 'P_95')
data_dirs[95] = os.path.join(data_root, 'Patch_ID_95')
input_files_rgb[95] = {'pC_col_2020_plant_PS495_Sun_smc_Krig.tif': ['Tempelberg1_soda3_06082020_transparent_mosaic_group1_merged_aligned_Patch_ID_95.tif']}
# input_files_rgb[95] = {'pC_col_2020_plant_PS495_Sun_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_95.tif',
#                                                                       'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_95.tif',
#                                                                       ]}
#
# ## Patch 105
# output_dirs[105] = os.path.join(output_root, 'P_105')
# data_dirs[105] = os.path.join(data_root, 'Patch_ID_105')
# input_files_rgb[105] = {'pC_col_2020_plant_PS4105_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_105.tif']}
# # input_files_rgb[105] = {'pC_col_2020_plant_PS4105_Pha_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_105.tif',
# #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_105.tif',
# #                                                                       ]}
#
# ## Patch 119
# output_dirs[119] = os.path.join(output_root, 'P_119')
# data_dirs[119] = os.path.join(data_root, 'Patch_ID_119')
# input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif']}
# # input_files_rgb[119] = {'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif',
# #                                                                       'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_119.tif',
# #                                                                       ]}
#
# #
# # # # Test for Lupine
# # output_dirs[39] = os.path.join(output_root, 'Lupine')
# # data_dirs['Lupine'] = os.path.join(data_root, 'Lupine')
# # input_files['Lupine'] = {
# #     'pC_col_2020_plant_PS4119_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_119.tif',
# #                                                   'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_119.tif'],
# #     'pC_col_2020_plant_PS459_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_59.tif',
# #                                                  'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_59.tif'],
# #     'pC_col_2020_plant_PS489_Lup_smc_Krig.tif': ['Tempelberg_sequ_11062020_dsm_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_11062020_index_ndvi_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_11062020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_16072020_dsm_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_16072020_index_ndvi_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_16072020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_17062020_dsm_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_17062020_index_ndvi_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_sequ_17062020_transparent_reflectance_merged_aligned_Patch_ID_89.tif',
# #                                                  'Tempelberg_Soda_22062020_transparent_mosaic_group1_merged_aligned_Patch_ID_89.tif']}
#

