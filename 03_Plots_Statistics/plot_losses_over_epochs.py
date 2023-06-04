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
output_root = '../../Output/shuffle/L2/'

# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_densenet_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_raytuning'
output_dirs = [
    # 'Patch_ID_73_RGB_baselinemodel_augmented_raytuning_nostopping',
    # 'P_65_resnet18_SCV_no_test_L1_ALB_TR1_E2000',
    # 'P_65_resnet18_SCV_no_test_TL-FC_L1_ALB_TR1_E2000',
    # 'P_65_resnet18_SCV_no_test_TL-Finetune_L1_ALB_TR1_E2000',
    # 'P_65_baselinemodel_SCV_no_test_L1_ALB_TR1_E2000',
    # 'P_68_resnet18_SCV_no_test_TL-FC_L1_ALB_TR1_E2000',
    # 'P_68_resnet18_SCV_no_test_SSL_ALB_E2000',
    # 'P_76_resnet18_SCV_no_test_L1_ALB_TR1_E1000',
    'P_68_resnet18_RCV_L2_cycle_E1000',
    # 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all_nonspatialCV',
              ]
# title = ['yield labels, spatial CV, no stopping', 'yield labels, spatial CV, early stopping', 'fake labels, spatial CV, early stopping', 'fake labels, non-spatial CV, no stopping' ]

output_dirs = [os.path.join(output_root, f) for f in output_dirs]

# SSL_suffix = '_domain-tuning'
# SSL_suffix = '_self-supervised'
SSL_suffix = ''
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

            if 'L2' in output_dirs[0]:
                df['val_loss'] = df['val_loss']**(1/2)
                df['train_loss'] = df['train_loss'] ** (1/2)
                ax = df.sort_values(by='epochs').plot(x='epochs', y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)])
                ax.set_ylabel('RMSE', fontsize=16)
            else:
                ax = df.sort_values(by='epochs').plot(x='epochs',y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)])
                ax.set_ylabel('MAE', fontsize=16)
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
            fig.savefig(os.path.join(output_dir, '{}training_statistics_f{}{}.png'.format(pref,k,SSL_suffix)))
