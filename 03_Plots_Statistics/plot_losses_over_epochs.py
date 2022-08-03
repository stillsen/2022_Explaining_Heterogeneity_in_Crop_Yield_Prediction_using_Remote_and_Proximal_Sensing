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
output_root = '../../Output/'

# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_densenet_augmented_fakelabels_tunedhyperparams_all'
# output_dir = '/beegfs/stiller/PatchCROP_all/Output/Patch_ID_73_RGB_baselinemodel_augmented_raytuning'
output_dirs = [
    # 'Patch_ID_73_RGB_baselinemodel_augmented_raytuning_nostopping',
    'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all_nonspatialCV',
    # 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all',
    # 'Patch_ID_73_RGB_baselinemodel_augmented_fakelabels_tunedhyperparams_all_nonspatialCV',
              ]
title = ['yield labels, spatial CV, no stopping', 'yield labels, spatial CV, early stopping', 'fake labels, spatial CV, early stopping', 'fake labels, non-spatial CV, no stopping' ]

output_dirs = [os.path.join(output_root, f) for f in output_dirs]

num_folds = 1
for i, output_dir in enumerate(output_dirs):
    for k in range(num_folds):
        if os.path.isfile(os.path.join(output_dir, 'training_statistics_f{}.csv'.format(k))):
            df = pd.read_csv(os.path.join(output_dir, 'training_statistics_f{}.csv'.format(k)))
            # df = pd.read_csv(os.path.join(output_dir, 'training_statistics_f{}.csv'.format(k)), names=['epochs', 'train_loss', 'val_loss', 'best_epoch'], header=)
            df.rename(columns={'Unnamed: 0' : 'epochs'}, inplace=True)
            # x = df['config.lr']
            # y = df['val_loss']
            # fig, ax = plt.subplots(1,1)
            ax = df.sort_values(by='epochs').plot(x='epochs', y=['val_loss', 'train_loss'])
            # ax.plot(x, y)
            fig = ax.get_figure()
            ax.set_title('Fold {}'.format(k), fontsize=16)
            ax.set_xlabel('epochs', fontsize=16)
            ax.set_ylabel('MSE loss', fontsize=16)

            # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
            #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
            # # ax.set_xlim([-100,100])
            # ax.set_ylim(lim)

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'training_statistics_f{}.png'.format(k)))
