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


def plot_pred_perf(y, y_hat, figure_path, file, ax, c='blue' , local_r2 = None):
    r = sklearn.metrics.r2_score(y_true=y, y_pred=y_hat)
    pc = np.corrcoef(y,y_hat)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # plt.scatter(y_hat, y, marker='.', c=c, alpha=0.1)
    ax.scatter(x=y_hat, y=y, marker='.', c=c, alpha=0.8)
    # ax.scatter(y_hat, y, marker='.', c=c, alpha=0.1)
    # plt.plot(plt.gca().get_xlim(), plt.gca().get_ylim(), ls="--", c=c)#c=".3")
    if local_r2 != None:
        # plt.title(r'$global R^{2} = $' + str(r)[:4] + r'$local R^{2} = $' + str(local_r2)[:4], fontsize=16)
        ax.set_title(r'$global R^{2} = $' + str(r)[:4] + r'$local R^{2} = $' + str(local_r2)[:4], fontsize=16)
    else:
        # plt.title(r'$R^{2} = $' + str(r)[:4], fontsize=16)
        ax.set_title(r'$R^{2} = $' + str(r)[:4]+r'$PC = $' + str(pc[0,1])[:4], fontsize=16)
    # plt.text(float(ax1.get_xlim()-10), float(ax1.get_ylim()-10), r'$R^{2} = $'+r)
    # plt.text(float(ax1.get_xlim()[1]*3/4), float(ax1.get_ylim()[1]*3/4))
    # plt.gca().set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
    # plt.gca().set_ylabel('Y [dt/ha]', fontsize=16)
    ax.set_xlabel(r'$\hat{Y}$ [dt/ha]', fontsize=16)
    ax.set_ylabel('Y [dt/ha]', fontsize=16)
    # ax1.set_xticks(range(30))
    # ax1.set_xticklabels(labels)
    # ax1.set_title('Yield')
    # ax1.yaxis.grid(True)
    # plt.savefig(os.path.join(figure_path, 'performance_'+file.split['.'][0]+'.png'))



output_dirs = dict()
data_dirs = dict()
input_files = dict()
input_files_rgb = dict()

# data_root = '/beegfs/stiller/PatchCROP_all/Data/'
data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
# output_root = '/beegfs/stiller/PatchCROP_all/Output/'
output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'
# output_root = '/media/stiller/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/'

output_dirs[65] = os.path.join(output_root, 'P_65_baselinemodel_SCV_no_test_L1_ALB_TR1_E2000')
# output_dirs[65] = os.path.join(output_root, 'P_65_resnet18_SCV_no_test_TL-Finetune_L1_ALB_TR1_E2000')
# output_dirs[65] = os.path.join(output_root, 'P_65_baselinemodel_SCV_no_test_L1_ALB_TR1_E2000')
# output_dirs[65] = os.path.join(output_root, 'P_65_resnet18_SCV_no_test_L1_ALB_TR1_E2000')
# output_dirs[65] = os.path.join(output_root, 'P_65_resnet18_SCV_no_test_TL-FC_L1_ALB_TR1_E2000')


# output_dirs[68] = os.path.join(output_root, 'P_68_baseline')
# output_dirs[68] = os.path.join(output_root, 'P_68_densenet')
# output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_L1_ALB_TR1_E1000')
# output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_TL-FC_L1_ALB_TR1_E2000')
# output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_SSL_ALB_E2000')
output_dirs[68] = os.path.join(output_root, 'P_68_resnet18_SCV_no_test_SSL_ALB_E2000')

# output_dirs[73] = os.path.join(output_root, 'P_73_baseline')
# output_dirs[73] = os.path.join(output_root, 'P_73_densenet')
output_dirs[73] = os.path.join(output_root, 'P_73_ssl')

# output_dirs[76] = os.path.join(output_root, 'P_76_baseline')
# output_dirs[76] = os.path.join(output_root, 'P_76_densenet')
# output_dirs[76] = os.path.join(output_root, 'P_76_resnet18_SCV_no_test_L1_ALB_TR1_E1000')
output_dirs[76] = os.path.join(output_root, 'P_76_resnet18_SCV_no_test_TL-FC_L1_ALB_TR1_E2000')



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

# patch_no = 73
patch_no = 65
# patch_no = 68
# patch_no = 76
# test_patch_no = 90

num_folds = 4

this_output_dir = output_dirs[patch_no]

# test_patch_name = 'Patch_ID_'+str(patch_no)+'_grn'
# test_patch_name = 'Patch_ID_'+str(patch_no)
test_patch_name = ''
### Combined Plots

sets = ['_train_',
        '_val_',
        '_test_',
        ]
# sets = [ '_test_']
# sets = ['_val_']
for s in sets:
    global_y = []
    global_y_hat = []
    local_cv = []
    local_corr = []
    # df_folds = []
    # df_global_r2 = []
    # df_local_r2_median = []
    # df_local_r2_mean = []
    # df_global_r = []
    # df_local_r_median = []
    # df_local_r_mean = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(num_folds):
        print('{}th-fold'.format(k))
        y = torch.load(os.path.join(this_output_dir, test_patch_name+'y' + s + str(k) + '.pt'))#.detach().numpy()
        # y = torch.load(os.path.join(this_output_dir, 'y' + s +  'domain-tuning.pt'))  # .detach().numpy()
        # y = torch.load(os.path.join(this_output_dir, 'y' + s + 'self-supervised.pt'))  # .detach().numpy()
        global_y.extend(y)
        y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat' + s + str(k) + '.pt'))#.detach().numpy()
        # y_hat = torch.load(os.path.join(this_output_dir, 'y_hat' + s + 'domain-tuning.pt'))  # .detach().numpy()
        # y_hat = torch.load(os.path.join(this_output_dir, 'y_hat' + s + 'self-supervised.pt'))  # .detach().numpy()

        global_y_hat.extend(y_hat)

        # if suff == '_train_':
        #     print(y)
        #     print(y_hat)

        local_cv.append(sklearn.metrics.r2_score(y, y_hat))
        local_corr.append(np.corrcoef(y, y_hat))

        # plot in combined figure
        plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax, file=test_patch_name+'y_yhat' + s + str(k) + '.png', c=colors[k])
        # # plot in separate figure
        # fig_k = plt.figure(k)
        # fig_k.clf()
        # ax_k = fig_k.add_subplot(111)
        # plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax_k, file='y_yhat' + s + str(k) + '.png', c=colors[k])
        # fig_k.savefig(os.path.join(this_output_dir, 'y_yhat_' + str(k) + '_' + s[:-1] + '.png'))
    local_r2 = np.mean(local_cv)
    # print('y: {}, y_hat: {}'.format(len(global_y), len(global_y_hat)))
    r = sklearn.metrics.r2_score(y_true=global_y, y_pred=global_y_hat)
    pc = np.corrcoef(global_y, global_y_hat)

    # plt.title('glob '+r'$R^{2} = $' + str(r)[:4] + ', loc ' +r'$R^{2} = $' + str(local_r2)[:4] +  ', median loc ' +r'$R^{2} = $' + str(np.median(local_cv))[:4] , fontsize=16)
    # ax.set_title('glob ' + r'$R^{2} = $' + str(r)[:4] + ', loc ' + r'$R^{2} = $' + str(local_r2)[:4] + ', median loc ' + r'$R^{2} = $' + str(np.median(local_cv))[:4], fontsize=16)

    if s == '_test_':
        ax.set_title('average ' + r'$R^{2} = $' + str(local_r2)[:4] + ', average r = ' + str(np.mean(local_corr))[:4], fontsize=16)
    else:
        ax.set_title('global ' + r'$R^{2} = $' + str(r)[:4] + ', local ' + r'$R^{2} = $' + str(local_r2)[:4] + '\n global r = ' + str(pc[0,1])[:4] + ', local r = ' + str(np.mean(local_corr))[:4], fontsize=16)
    ###
    df = pd.DataFrame({'global_r2': [str(r)[:4]],
                       'local_r2_median': [np.median(local_cv)],
                       'local_r2_mean': [str(local_r2)[:4]],
                       'global_r':[str(pc[0,1])[:4]],
                       'local_r_median':[str(np.median(local_corr))[:4]],
                       'local_r_mean':[str(np.mean(local_corr))[:4]],
                       # 'lo': run_time,
                       # 'ft_val_loss':ft_val_losses,
                       # 'ft_train_loss':ft_train_losses,
                       # 'ft_best_epoch':ft_best_epoch,
                       })
    df.to_csv(os.path.join(this_output_dir, 'performance_metrics_f' + str(k) + '_' + s + '.csv'), encoding='utf-8')
    ###
    # ax.set_title('glob ' + r'$R^{2} = $' + str(r)[:4] + ', loc ' + r'$R^{2} = $' + str(local_r2)[:4] + ', median loc ' + r'$R^{2} = $' + str(np.median(local_cv))[:4], fontsize=16)
    print('global r: %s, local r: %s, SD: %s, min: %s, max: %s, median %s' %(str(r)[:4], str(local_r2)[:4], str(np.std(local_cv))[:4], str(np.min(local_cv))[:4], str(np.max(local_cv)), str(np.median(local_cv))[:4]))
    # plt.tight_layout()
    fig.tight_layout()
    lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
           max(ax.get_ylim()[1], ax.get_xlim()[1])]
    # ax.set_xlim([-100,100])
    ax.set_ylim(lim)
    # fig.savefig(os.path.join(this_output_dir, 'y_yhat_global' + s[:-1] + 'self-supervised.png'))
    # fig.savefig(os.path.join(this_output_dir, 'y_yhat_global' + s[:-1] + 'domain-tuning.png'))
    fig.savefig(os.path.join(this_output_dir, test_patch_name+'y_yhat_global' + s[:-1] + '.png'))

### Single Plots
    # lim = [min(plt.gca().get_ylim()[0], plt.gca().get_xlim()[0]),
    #        max(plt.gca().get_ylim()[1], plt.gca().get_xlim()[1])]
    # plt.gca().set_xlim(lim)
    # plt.gca().set_ylim(lim)
    # plt.savefig(os.path.join(this_output_dir, 'y_yhat_global' + suff[:-1] + '.png'))

    for s in sets:
        global_y = []
        global_y_hat = []
        local_cv = []
        local_corr = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in range(num_folds):
            print('{}th-fold'.format(k))
            y = torch.load(os.path.join(this_output_dir, test_patch_name+'y' + s + str(k) + '.pt'))  # .detach().numpy()
            # y = torch.load(os.path.join(this_output_dir, 'y' + s + 'domain-tuning.pt'))  # .detach().numpy()
            # y = torch.load(os.path.join(this_output_dir, 'y' + s + 'self-supervised.pt'))  # .detach().numpy()
            global_y.extend(y)
            y_hat = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat' + s + str(k) + '.pt'))  # .detach().numpy()
            # y_hat = torch.load(os.path.join(this_output_dir, 'y_hat' + s + 'domain-tuning.pt'))  # .detach().numpy()
            # y_hat = torch.load(os.path.join(this_output_dir, 'y_hat' + s + 'self-supervised.pt'))  # .detach().numpy()

            global_y_hat.extend(y_hat)

            # if suff == '_train_':
            #     print(y)
            #     print(y_hat)

            local_cv.append(sklearn.metrics.r2_score(y, y_hat))
            local_corr.append(np.corrcoef(y,y_hat))
            # plot in combined figure
            # plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax, file='y_yhat' + s + str(k) + 'self-supervised.png', c=colors[k])
            # plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax, file='y_yhat' + s + str(k) + 'domain-tuning.png', c=colors[k])
            plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax, file=test_patch_name+'y_yhat' + s + str(k) + '.png', c=colors[k])
            # plot in separate figure
            fig_k = plt.figure(k)
            fig_k.clf()
            ax_k = fig_k.add_subplot(111)
            # plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax_k, file='y_yhat' + s + str(k) + 'self-supervised.png', c=colors[k])
            # plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax_k, file='y_yhat' + s + str(k) + 'domain-tuning.png', c=colors[k])
            plot_pred_perf(y=y, y_hat=y_hat, figure_path=this_output_dir, ax=ax_k, file=test_patch_name+'y_yhat' + s + str(k) + '.png', c=colors[k])
            # fig_k.savefig(os.path.join(this_output_dir, 'y_yhat_' + str(k) + '_' + s[:-1] + 'self-supervised.png'))
            # fig_k.savefig(os.path.join(this_output_dir, 'y_yhat_' + str(k) + '_' + s[:-1] + 'domain-tuning.png'))
            fig_k.savefig(os.path.join(this_output_dir, test_patch_name+'y_yhat_' + str(k) + '_' + s[:-1] + '.png'))
        local_r2 = np.mean(local_cv)
        # print('y: {}, y_hat: {}'.format(len(global_y), len(global_y_hat)))
        r = sklearn.metrics.r2_score(y_true=global_y, y_pred=global_y_hat)
        pc = np.corrcoef(global_y,global_y_hat)

        # plt.title('glob '+r'$R^{2} = $' + str(r)[:4] + ', loc ' +r'$R^{2} = $' + str(local_r2)[:4] +  ', median loc ' +r'$R^{2} = $' + str(np.median(local_cv))[:4] , fontsize=16)
        # ax.set_title('glob ' + r'$R^{2} = $' + str(r)[:4] + ', loc ' + r'$R^{2} = $' + str(local_r2)[:4] + ', median loc ' + r'$R^{2} = $' + str(np.median(local_cv))[:4], fontsize=16)
        if s == '_test_':
            ax.set_title('average ' + r'$R^{2} = $' + str(local_r2)[:4] + ', average r = ' + str(np.mean(local_corr))[:4], fontsize=16)
        else:
            ax.set_title('global ' + r'$R^{2} = $' + str(r)[:4] + ', local ' + r'$R^{2} = $' + str(local_r2)[:4] + '\n global r = ' + str(pc[0,1])[:4] + ', local r = ' + str(np.mean(local_corr))[:4], fontsize=16)
        print('global r: %s, local r: %s, SD: %s, min: %s, max: %s, median %s' % (str(r)[:4], str(local_r2)[:4], str(np.std(local_cv))[:4], str(np.min(local_cv))[:4], str(np.max(local_cv)), str(np.median(local_cv))[:4]))
        # plt.tight_layout()
        # fig.tight_layout()
        # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
        #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
        # # ax.set_xlim([-100,100])
        # ax.set_ylim(lim)
        # fig.savefig(os.path.join(this_output_dir, 'y_yhat_global' + s[:-1] + '.png'))
        #
        # lim = [min(plt.gca().get_ylim()[0], plt.gca().get_xlim()[0]),
        #        max(plt.gca().get_ylim()[1], plt.gca().get_xlim()[1])]
        # plt.gca().set_xlim(lim)
        # plt.gca().set_ylim(lim)
        # plt.savefig(os.path.join(this_output_dir, 'y_yhat_global' + s[:-1] + '.png'))