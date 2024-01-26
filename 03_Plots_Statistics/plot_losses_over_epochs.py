import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from ray.util import inspect_serializability
from ray.tune import ExperimentAnalysis

# output_root = '/beegfs/stiller/PatchCROP_all/Output/'
# output_root = '../../Output/shuffle/L2/'
output_root = '../../Output/SSL/Tests/'
# output_root = '../../Output/ks/'

# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_densenet_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_raytuning'
output_dirs = [
    # 'Combined_crops_train_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224',
    # 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224-black-added-to-train-and-val',
    # 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224-black-added-to-train',
    # 'P_68_VICRegLConvNext_SCV_no_test_E100lightly-VICRegLConvNext_kernel_size_224_recompute',
    'P_68_VICRegLConvNext_SCV_no_test_E10000lightly-VICRegLConvNext_kernel_size_224_heating',

              ]
# title = ['yield labels, spatial CV, no stopping', 'yield labels, spatial CV, early stopping', 'fake labels, spatial CV, early stopping', 'fake labels, non-spatial CV, no stopping' ]

output_dirs = [os.path.join(output_root, f) for f in output_dirs]

SSL_suffix = '_domain-tuning'
# SSL_suffix = '_self-supervised'
# SSL_suffix = ''
num_folds = 4
for i, output_dir in enumerate(output_dirs):
    for k in range(num_folds):
        if os.path.isfile(os.path.join(output_dir, 'training_statistics_f{}{}.csv'.format(k,SSL_suffix))):
            df = pd.read_csv(os.path.join(output_dir, 'training_statistics_f{}{}.csv'.format(k,SSL_suffix)))
            # df = pd.read_csv(os.path.join(output_dir, 'training_statistics_f{}.csv'.format(k)), names=['epochs', 'train_loss', 'val_loss', 'best_epoch'], header=)
            df.rename(columns={'Unnamed: 0' : 'epochs'}, inplace=True)
            # x = df['config.lr']
            # y = df['val_loss']
            # fig, ax = plt.subplots(1,1)
            # ax = df.sort_values(by='epochs').plot(x='epochs', y=['val_loss', 'train_loss'])
            pref = ''
            # pref = 'ss_'
            # pref = 'domain_'

            # RMSE
            # if 'L2' in output_dirs[0]:
            df['val_loss'] = df['val_loss']**(1/2)
            df['train_loss'] = df['train_loss'] ** (1/2)
            ax = df.sort_values(by='epochs').plot(x='epochs', y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)])
            if SSL_suffix == '' or SSL_suffix == '_domain-tuning':
                ax.set_ylabel('RMSE', fontsize=16)
            elif SSL_suffix == '_self-supervised':
                ax.set_ylabel('VICReg Loss', fontsize=16)
            # MAE?
            # ax = df.sort_values(by='epochs').plot(x='epochs',y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)])
            # ax.set_ylabel('MAE', fontsize=16)

            # plot line at best epoch
            ax.axvline(x=df['best_epoch'][0], color='red', linestyle='dashed')
            ax.text(x=df['best_epoch'][0]-3, y=ax.get_ylim()[1]*1.01, s=str(df['best_epoch'][0]), color='red' )

            # ax.plot(x, y)
            fig = ax.get_figure()
            # ax.set_title('Fold {}'.format(k), fontsize=16)
            ax.set_xlabel('epochs', fontsize=16)
            # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
            #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
            # # ax.set_xlim([-100,100])
            # ax.set_ylim([0, 100])

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, '{}training_statistics_f{}{}.png'.format(pref,k,SSL_suffix)), dpi=400)

        if os.path.isfile(os.path.join(output_dir, 'lrs_f{}{}.csv'.format(k,SSL_suffix))):
            df = pd.read_csv(os.path.join(output_dir, 'lrs_f{}{}.csv'.format(k,SSL_suffix)))
            df.rename(columns={'Unnamed: 0' : 'epochs'}, inplace=True)

            pref = ''

            df['lrs'] = df['lrs'].str.strip('[]').astype(float)
            ax = df.sort_values(by='epochs').plot(x='epochs', y=['{}lrs'.format(pref)])
            ax.set_ylabel('lr', fontsize=16)

            # ax.plot(x, y)
            fig = ax.get_figure()
            # ax.set_title('Fold {}'.format(k), fontsize=16)
            ax.set_xlabel('epochs', fontsize=16)
            # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
            #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
            # # ax.set_xlim([-100,100])
            # ax.set_ylim([0, 100])

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, '{}lrs_f{}{}.png'.format(pref,k,SSL_suffix)), dpi=400)