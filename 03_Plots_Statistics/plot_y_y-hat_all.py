# -*- coding: utf-8 -*-
"""
y-y_hat plot for train, val and test set preditions
"""

# Built-in/Generic Imports

# Libs
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import sklearn.metrics
import pandas as pd

# Own modules


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'



output_dirs = dict()
data_dirs = dict()
input_files = dict()
input_files_rgb = dict()

# data_root = '/beegfs/stiller/PatchCROP_all/Data/'
data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
# output_root = '/beegfs/stiller/PatchCROP_all/Output/'
# output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'
output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/shuffle/L2/'
# output_root = '/media/stiller/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

output_dirs = [
    'P_68_resnet18_SCV_RCV',
    'P_65_resnet18_SCV_RCV',
    'P_76_resnet18_SCV_RCV',
    # 'P_68_baselinemodel_SCV_RCV',
    # 'P_65_baselinemodel_SCV_RCV',
    # 'P_76_baselinemodel_SCV_RCV',
              ]
output_dirs = [os.path.join(output_root, f) for f in output_dirs]
# figure_path='/home/stillsen/Documents/GeoData/PatchCROP/AIA/Figures'

ctype= ['0','1','2','3','4','5','6','7','8']
taxonomic_groups_to_color = {'0': 1/10, '1': 2/10, '2': 3/10,
                             '3': 4/10,
                             '4': 5/10, '5': 6/10, '6':7/10, '7':8/10, '8':9/10}
cc = [taxonomic_groups_to_color[x] for x in ctype]
# color map
cmap = plt.cm.get_cmap('tab10', 9)
# #encode cc in cmap
colors = cmap(cc)



# patch_no = 73
patch_no = 76
# patch_no = 65
# patch_no = 68
# test_patch_no = 90

num_folds = 4

# test_patch_name = 'Patch_ID_'+str(patch_no)+'_grn'
# test_patch_name = 'Patch_ID_'+str(patch_no)
test_patch_name = ''

sets = ['_train_',
        '_val_',
        '_test_',
        ]

i = 0
legend = True

font_size = 15
medium_font_font_size = 13
small_font_font_size = 10
plt.rc('font', size=medium_font_font_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_font_font_size)     # fontsize of the axes title
plt.rc('axes', labelsize=small_font_font_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_font_font_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_font_font_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium_font_font_size)    # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

# fig, axes = plt.subplots(6,3, figsize=(9, 15),layout="constrained")
fig, axes = plt.subplots(6,3, figsize=(9, 15))

# spacing
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots
left  = 0.1  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.05  # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.25   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# fig.set_size_inches(11, 16, forward=True)


crop_type = [r'$\bf{Maize}$',
             r'$\bf{Soy}$',
             r'$\bf{Sunflowers}$']
for this_output_dir in output_dirs:
    for cv in ['RCV_', 'SCV_']:
        j = 0
        for s in sets:
        ### Combined Plots - all folds in one figure
            # pool all y over all folds
            global_y = []
            # pool all y_hat over all folds
            global_y_hat = []
            # list containing all R2 values for each fold
            local_r2 = []
            # list containing all r values for each fold
            local_r = []
            # average performance on test_set
            test_r2 = None
            test_r = None
            y_hat = None
            if s == '_test_':
                '''
                test set performance is the performance of the averaged model prediction, i.e. mean prediction values
                '''
                for k in range(num_folds):
                    print('combined plots: {}th-fold'.format(k))
                    # load y
                    y = torch.load( os.path.join(this_output_dir, test_patch_name + 'y' + s + cv + str(k) + '.pt'))
                    # load y_hat and add y_hat of successive folds (later divide by num_folds for average prediction score)
                    if k == 0:
                        y_hat = torch.load(os.path.join(this_output_dir, test_patch_name + 'y_hat' + s + cv + str(k) + '.pt'))
                    else:
                        y_hat = [y_hat[i]+prediction for i, prediction in enumerate(torch.load(os.path.join(this_output_dir, test_patch_name + 'y_hat' + s + cv + str(k) + '.pt')))]

                y_hat = [prediction/4 for prediction in y_hat]
                # plot in combined figure
                axes[i][j].scatter(x=y_hat, y=y, marker='.', c=colors[5], alpha=0.8)

                test_r2 = np.mean(sklearn.metrics.r2_score(y, y_hat))
                test_r = np.corrcoef(y, y_hat)[0,1]

                ### save performance metrics
                df = pd.DataFrame({
                                   'average_r2': [str(test_r2)[:4]],
                                   'average_r': [str(test_r)[:4]],
                                   })
                df.to_csv(os.path.join(this_output_dir, 'performance_metrics_f' + str(k) + '_' + s + cv + '.csv'), encoding='utf-8')
                ###

            else: # val and train
                '''
                local performance score: averaged performance score across folds
                global performance score: performance score over all predictions
                '''

                for k in range(num_folds):
                    print('combined plots: {}th-fold'.format(k))
                    # load y and y_hat
                    y = torch.load(os.path.join(this_output_dir, test_patch_name+'y' + s + cv + str(k) + '.pt'))
                    y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat' + s  + cv + str(k) + '.pt'))
                    # pool y and y_hat globally
                    global_y.extend(y)
                    global_y_hat.extend(y_hat)
                    # save local prediction performances y and y_hat globally
                    r2 = sklearn.metrics.r2_score(y, y_hat)
                    r = np.corrcoef(y, y_hat)[0,1]
                    local_r2.append(r2)
                    local_r.append(r)

                    # plot in combined figure
                    axes[i][j].scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)

                avg_local_r2 = np.mean(local_r2)
                glogal_r2 = sklearn.metrics.r2_score(y_true=global_y, y_pred=global_y_hat)
                global_r = np.corrcoef(global_y, global_y_hat)[0,1]

                ### save performance metrics
                df = pd.DataFrame({'global_r2': [str(glogal_r2)[:4]],
                                   'local_r2_median': [np.median(local_r2)],
                                   'local_r2_mean': [str(avg_local_r2)[:4]],
                                   'global_r': [str(global_r)[:4]],
                                   'local_r_median': [str(np.median(local_r))[:4]],
                                   'local_r_mean': [str(np.mean(local_r))[:4]],
                                   })
                df.to_csv(os.path.join(this_output_dir, 'performance_metrics_f' + str(k) + '_' + s + '.csv'), encoding='utf-8')
            # text
            if i == 0:
                if j == 0:
                    axes[i][j].set_title(r'$\bf{Train}$')
                if j == 1:
                    axes[i][j].set_title(r'$\bf{Validation}$')
                if j == 2:
                    axes[i][j].set_title(r'$\bf{Test}$')
            if i==5:
                axes[i][j].set_xlabel('predicted yield [dt/ha]')
            if j == 0:
                # axes[i][j].set_ylabel('yield [dt/ha]')
                axes[i][j].set_ylabel('')
                axes[i][j].text(-0.18, 0.5, 'yield [dt/ha]', rotation=90, va='center', transform=axes[i][j].transAxes, fontsize=small_font_font_size)
                if i%2 == 0:
                    # axes[i][j].text(-0.13, 0.95, r'$\bf{RCV}$', fontsize=font_size, transform=axes[i][j].transAxes)
                    axes[i][j].text(-0.3, 0.45, r'$\it{RCV}$', rotation=90, va='center',  transform=axes[i][j].transAxes)
                else:
                    # axes[i][j].text(-0.13, 0.95, r'$\bf{SCV}$', fontsize=font_size, transform=axes[i][j].transAxes)
                    axes[i][j].text(-0.3, 0.45, r'$\it{SCV}$', rotation=90, va='center', transform=axes[i][j].transAxes)
            if j < 2 : # plot train and validation pearson's r
                axes[i][j].text(0.025, 0.65 ,'\nglobal r = ' + str(global_r)[:4] + '\nlocal median r = ' + str(np.median(local_r))[:4], transform=axes[i][j].transAxes)
            else: # test mean r
                axes[i][j].text(0.025, 0.65, 'mean prediction\nr = ' + str(test_r)[:4], transform=axes[i][j].transAxes)


            # group line and crop name
            if i % 2 == 0 and j==0:
                axes[i][j].text(-0.4, 0, crop_type[int(i/2)], rotation=90, va='center', transform=axes[i][j].transAxes)
                # axes[i][j].plot([-0.2, -0.2], [-1, 1], color='black', transform=axes[i][j].transAxes)
                axes[i][j].plot([-0.22, -0.22], [-1.28, 1], color='black', clip_on=False, transform=axes[i][j].transAxes)


            lim = [min(axes[i][j].get_ylim()[0], axes[i][j].get_xlim()[0]),
                   max(axes[i][j].get_ylim()[1], axes[i][j].get_xlim()[1])]
            # axes[i][j].set_ylim(lim)

            j += 1
        i += 1
# fig.tight_layout()
plt.show()
#
fig.savefig(os.path.join(output_root, 'y_yhat_resnet_all.svg'), papertype='a4')
fig.savefig(os.path.join(output_root, 'y_yhat_resnet_all.png'), papertype='a4')

# fig.savefig(os.path.join(output_root, 'y_yhat_baseline_all.svg'), papertype='a4')
# fig.savefig(os.path.join(output_root, 'y_yhat_baseline_all.png'), papertype='a4')