# -*- coding: utf-8 -*-
"""
y-y_hat plot for train, val and test set preditions
"""

# Built-in/Generic Imports

# Libs
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import torch
import sklearn.metrics

# Own modules


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'


# def plot_pred_perf(y, y_hat, figure_path, file, c='blue' , local_r2 = None):
#     r = sklearn.metrics.r2_score(y, y_hat)
#     # fig = plt.figure()
#     # ax1 = fig.add_subplot(111)
#     plt.scatter(y_hat, y, marker='.', c=c, alpha=0.1)
#     # plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), ls="--", c=c)#c=".3")
#     if local_r2 != None:
#         plt.title(r'$global R^{2} = $' + str(r)[:4] + r'$local R^{2} = $' + str(local_r2)[:4], fontsize=16)
#     else:
#         plt.title(r'$R^{2} = $' + str(r)[:4], fontsize=16)
#     # plt.text(float(ax1.get_xlim()-10), float(ax1.get_ylim()-10), r'$R^{2} = $'+r)
#     # plt.text(float(ax1.get_xlim()[1]*3/4), float(ax1.get_ylim()[1]*3/4))
#     plt.gca().set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
#     plt.gca().set_ylabel('Y [dt/ha]', fontsize=16)
#     # ax1.set_xticks(range(30))
#     # ax1.set_xticklabels(labels)
#     # ax1.set_title('Yield')
#     # ax1.yaxis.grid(True)
#     # plt.savefig(os.path.join(figure_path, 'performance_'+file.split['.'][0]+'.png'))
#


output_dirs = dict()
data_dirs = dict()
input_files = dict()
input_files_rgb = dict()

# data_root = '/beegfs/stiller/PatchCROP_all/Data/'
data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
# output_root = '/beegfs/stiller/PatchCROP_all/Output/'
# output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'
output_root = '/media/stiller/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

## Patch 73
# output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_not_augmented_custom_')
# output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB+_densenet_augmented_custom')
# output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_densenet_augmented_custom_btf')
output_dirs[73] = os.path.join(output_root, 'Patch_ID_73_RGB_baselinemodel_augmented_origlabels_test_nsp')
output_dirs[119] = os.path.join(output_root, 'Patch_ID_119_RGB_densenet_augmented')
output_dirs[105] = os.path.join(output_root, 'Patch_ID_105_RGB_densenet_augmented')
output_dirs[19] = os.path.join(output_root, 'Patch_ID_19_RGB_densenet_augmented')
output_dirs[50] = os.path.join(output_root, 'Patch_ID_50_RGB_densenet_augmented')
output_dirs[95] = os.path.join(output_root, 'Patch_ID_95_RGB_densenet_augmented')

# figure_path='/home/stillsen/Documents/GeoData/PatchCROP/AIA/Figures'

ctype= ['0','1','2','3','4','5','6','7','8']
taxonomic_groups_to_color = {'0': 1/10, '1': 2/10, '2': 3/10,
                             '3': 4/10,
                             '4': 5/10, '5': 6/10, '6':7/10, '7':8/10, '8':9/10}
cc = [taxonomic_groups_to_color[x] for x in ctype]
# colormap = {'Lup': 'black', 'Pha': 'red', 'Sun' : 'blue', 'SOats': 'green', 'Soy': 'yellow', 'Maiz': 'purple'}
# colors = [colormap[x] for x in ctype]
# #receive color map
# cmap = plt.cm.get_cmap('Dark2', 6)
cmap = plt.cm.get_cmap('tab10', 9)
# #encode cc in cmap
colors = cmap(cc)

this_output_dir = output_dirs[73]

for suff in ['val_', 'train_']:
# for suff in ['train_']:
    plt.figure()
    for k in range(1):
        df = pd.read_csv(os.path.join(this_output_dir, 'training_statistics_f'+str(k)+'.csv'))#.detach().numpy()
        plt.plot(df[suff+'loss'], c=colors[k])

    plt.title(suff+'error for each fold' , fontsize=16)
    plt.gca().set_ylabel('MSE', fontsize=16)
    plt.gca().set_xlabel('Epoch', fontsize=16)
    plt.tight_layout()
    # lim = [min(plt.gca().get_ylim()[0], plt.gca().get_xlim()[0]),
    #        max(plt.gca().get_ylim()[1], plt.gca().get_xlim()[1])]
    # plt.gca().set_xlim(lim)
    # if suff == 'train_':
    #     plt.gca().set_ylim([0,70])
    # else:
    #     pass
    #     plt.gca().set_ylim([0, 400])
    # plt.show()
    plt.savefig(os.path.join(this_output_dir, suff+'training_statistics.png'))
    del df