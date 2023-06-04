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

output_dirs[65] = os.path.join(output_root, 'P_65_resnet18_SCV_no_test_L2_cycle_E1000')
output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_L2_cycle_E1000')
# output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_SSL_ALB_E2000')
output_dirs[73] = os.path.join(output_root, 'P_73_ssl')
output_dirs[76] = os.path.join(output_root, 'P_76_baselinemodel_RCV_L2_cycle_E1000')

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

# colors = ['#78C850',
#           '#F08030',
#           '#6890F0',
#           '#A8B820',
#           '#F8D030',
#           '#E0C068',
#           '#C03028',
#           '#F85888',
#           '#98D8D8']

# patch_no = 73
# patch_no = 76
patch_no = 68
# patch_no = 68
# test_patch_no = 90

num_folds = 4

this_output_dir = output_dirs[patch_no]

# test_patch_name = 'Patch_ID_'+str(patch_no)+'_grn'
# test_patch_name = 'Patch_ID_'+str(patch_no)
test_patch_name = ''

sets = ['_train_',
        '_val_',
        '_test_',
        ]

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
    if s == '_test_':
        '''
        test set performance is the performance of the averaged model prediction, i.e. mean prediction values
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        y_hat = None

        for k in range(num_folds):
            print('combined plots: {}th-fold'.format(k))
            # load y
            y = torch.load( os.path.join(this_output_dir, test_patch_name + 'y' + s + str(k) + '.pt'))
            # load y_hat and add y_hat of successive folds (later divide by num_folds for average prediction score)
            if k == 0:
                y_hat = torch.load(os.path.join(this_output_dir, test_patch_name + 'y_hat' + s + str(k) + '.pt'))
            else:
                y_hat = [y_hat[i]+prediction for i, prediction in enumerate(torch.load(os.path.join(this_output_dir, test_patch_name + 'y_hat' + s + str(k) + '.pt')))]

        y_hat = [prediction/4 for prediction in y_hat]
        # plot in combined figure
        # ax.scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)
        ax.scatter(x=y_hat, y=y, marker='.', c=colors[5], alpha=0.8)

        test_r2 = np.mean(sklearn.metrics.r2_score(y, y_hat))
        test_r = np.corrcoef(y, y_hat)[0,1]

        ax.set_title('average ' + r'$R^{2} = $' + str(test_r2)[:4] + ', average r = ' + str(test_r)[:4], fontsize=16)

        ax.set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
        ax.set_ylabel('Y [dt/ha]', fontsize=16)

        fig.tight_layout()
        # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
        #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
        # ax.set_xlim([-100,100])
        # ax.set_ylim(lim)

        fig.savefig(os.path.join(this_output_dir, test_patch_name + 'y_yhat_average' + s[:-1] + '.png'))

        ### save performance metrics
        df = pd.DataFrame({
                           'average_r2': [str(test_r2)[:4]],
                           'average_r': [str(test_r)[:4]],
                           })
        df.to_csv(os.path.join(this_output_dir, 'performance_metrics_f' + str(k) + '_' + s + '.csv'), encoding='utf-8')
        ###

    else: # val and train
        '''
        local performance score: averaged performance score across folds
        global performance score: performance score over all predictions
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in range(num_folds):
            print('combined plots: {}th-fold'.format(k))
            # load y and y_hat
            y = torch.load(os.path.join(this_output_dir, test_patch_name+'y' + s + str(k) + '.pt'))
            y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat' + s + str(k) + '.pt'))
            # pool y and y_hat globally
            global_y.extend(y)
            global_y_hat.extend(y_hat)
            # save local prediction performances y and y_hat globally
            r2 = sklearn.metrics.r2_score(y, y_hat)
            r = np.corrcoef(y, y_hat)[0,1]
            local_r2.append(r2)
            local_r.append(r)

            # plot in combined figure
            ax.scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)

        avg_local_r2 = np.mean(local_r2)
        glogal_r2 = sklearn.metrics.r2_score(y_true=global_y, y_pred=global_y_hat)
        global_r = np.corrcoef(global_y, global_y_hat)[0,1]

        ax.set_title('global ' + r'$R^{2} = $' + str(glogal_r2)[:4] + ', global r = ' + str(global_r)[:4]+ '\n local ' + r'$R^{2} = $' + str(avg_local_r2)[:4]  + ', local r = ' + str(np.mean(local_r))[:4], fontsize=16)
        ax.set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
        ax.set_ylabel('Y [dt/ha]', fontsize=16)

        fig.tight_layout()
        lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
               max(ax.get_ylim()[1], ax.get_xlim()[1])]
        # ax.set_xlim([-100,100])
        ax.set_ylim(lim)

        fig.savefig(os.path.join(this_output_dir, test_patch_name + 'y_yhat_global' + s[:-1] + '.png'))

    ### save performance metrics
        df = pd.DataFrame({'global_r2': [str(glogal_r2)[:4]],
                           'local_r2_median': [np.median(local_r2)],
                           'local_r2_mean': [str(avg_local_r2)[:4]],
                           'global_r':[str(global_r)[:4]],
                           'local_r_median':[str(np.median(local_r))[:4]],
                           'local_r_mean':[str(np.mean(local_r))[:4]],
                           })
        df.to_csv(os.path.join(this_output_dir, 'performance_metrics_f' + str(k) + '_' + s + '.csv'), encoding='utf-8')
        ###

### Single Plots
    for k in range(num_folds):
        print('single plots: {}th-fold'.format(k))
        y = torch.load(os.path.join(this_output_dir, test_patch_name+'y' + s + str(k) + '.pt'))

        y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat' + s + str(k) + '.pt'))

        local_r2 = sklearn.metrics.r2_score(y, y_hat)
        local_r = np.corrcoef(y, y_hat)[0,1]

        # plot in separate figure
        fig = plt.figure(k)
        fig.clf()
        fig.tight_layout()
        ax = fig.add_subplot(111)

        ax.scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)

        ax.set_title('local ' + r'$R^{2} = $' + str(local_r2)[:4] + ', local r = ' + str(local_r)[:4], fontsize=16)
        ax.set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
        ax.set_ylabel('Y [dt/ha]', fontsize=16)


        lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
               max(ax.get_ylim()[1], ax.get_xlim()[1])]
        # ax.set_xlim([-100,100])
        ax.set_ylim(lim)

        fig.savefig(os.path.join(this_output_dir, test_patch_name+'y_yhat_' + str(k) + '_' + s[:-1] + '.png'))
