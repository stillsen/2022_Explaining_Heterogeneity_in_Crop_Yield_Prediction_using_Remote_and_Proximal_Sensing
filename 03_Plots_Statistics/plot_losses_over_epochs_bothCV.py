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
    'P_68_resnet18_SCV_RCV',
    # 'P_65_resnet18_SCV_RCV',
    # 'P_76_resnet18_SCV_RCV',
    # 'P_68_baselinemodel_SCV_RCV',
    # 'P_65_baselinemodel_SCV_RCV',
    # 'P_76_baselinemodel_SCV_RCV',
              ]
output_dirs = [os.path.join(output_root, f) for f in output_dirs]

# SSL_suffix = '_domain-tuning'
# SSL_suffix = '_self-supervised'
SSL_suffix = ''
num_folds = 4

font_size = 15
medium_font_font_size = 13
small_font_font_size = 10
plt.rc('font', size=medium_font_font_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_font_font_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_font_font_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_font_font_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_font_font_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium_font_font_size)    # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

fig, axes = plt.subplots(2,4, figsize=(9, 6))

# spacing
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.2   # the amount of height reserved for white space between subplots
left  = 0.1  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.25   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

y = 0
legend = True

for cv in ['_RCV', '_SCV']:
    x = 0
    for i, output_dir in enumerate(output_dirs):
        for k in range(num_folds):
            if os.path.isfile(os.path.join(output_dir, 'training_statistics{}_f{}{}.csv'.format(cv,k,SSL_suffix))):
                df = pd.read_csv(os.path.join(output_dir, 'training_statistics{}_f{}{}.csv'.format(cv,k,SSL_suffix)))

                df.rename(columns={'Unnamed: 0' : 'epochs'}, inplace=True)
                # x = df['config.lr']
                # y = df['val_loss']
                # fig, ax = plt.subplots(1,1)
                # ax = df.sort_values(by='epochs').plot(x='epochs', y=['val_loss', 'train_loss'])
                pref = ''
                # pref = 'ss_'
                # pref = 'domain_'

                if 'L2' in output_dirs[0]:
                    df['val_loss'] = (df['val_loss']**(1/2))*0.1 # times 0.1 to convert from dt/ha to t/ha
                    df['train_loss'] = (df['train_loss'] ** (1/2))*0.1
                    axes[y][x] = df.sort_values(by='epochs').plot(x='epochs', y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)], ax=axes[y][x], legend=legend)
                    if x == 0:
                        axes[y][x].set_ylabel('RMSE')
                else:
                    axes[y][x] = df.sort_values(by='epochs').plot(x='epochs',y=['{}val_loss'.format(pref), '{}train_loss'.format(pref)], ax=axes[y][x], legend=legend)
                    if x==0:
                        axes[y][x].set_ylabel('MAE')
                # plot line at best epoch
                axes[y][x].axvline(x=df['best_epoch'][0], color='red', linestyle='dashed')
                axes[y][x].text(x=df['best_epoch'][0]-3, y=axes[y][x].get_ylim()[1]*1.01, s=str(df['best_epoch'][0]), color='red')


                if cv == '_SCV':
                    axes[y][x].set_xlabel('epochs')
                    if x==0:
                        axes[y][x].text(-0.25, 1.05, r'$\bf{SCV}$', va='center', transform=axes[y][x].transAxes)
                        # axes[y][x].text(-160, 8.6, r'$\bf{SCV}$', va='center')  # Maize base
                        # axes[y][x].text(-160, 8.6, r'$\bf{SCV}$', va='center') # Maize resnet
                        # axes[y][x].text(-160, 1.85, r'$\bf{SCV}$', va='center') # Soy base
                        # axes[y][x].text(-160, 2.3, r'$\bf{SCV}$', va='center') # Soy resnet
                        # axes[y][x].text(-160, 10.5, r'$\bf{SCV}$', va='center') # Sunflowers base
                        # axes[y][x].text(-160, 15.5, r'$\bf{SCV}$', va='center') # Sunflowers base
                        # axes[y][x].set_subtitle('SCV')

                else:
                    axes[y][x].set_title('fold {}'.format(k+1))
                    axes[y][x].set_xlabel('')
                    if x==0:
                        axes[y][x].text(-0.25, 1.05, r'$\bf{RCV}$', va='center', transform=axes[y][x].transAxes)  #
                        # axes[y][x].text(-160, 8.1, r'$\bf{RCV}$', va='center')  # Maize base
                        # axes[y][x].text(-160, 7.6, r'$\bf{RCV}$', va='center') # Maize resnet
                        # axes[y][x].text(-160, 1.8, r'$\bf{RCV}$', va='center') # Soy base
                        # axes[y][x].text(-160, 1.55, r'$\bf{RCV}$', va='center') # Soy resnet
                        # axes[y][x].text(-160, 10.5, r'$\bf{RCV}$', va='center') # Sunflowers base
                        # axes[y][x].text(-160, 7.25, r'$\bf{RCV}$', va='center') # Sunflowers base
                # lim = [min(ax.get_ylim()[0], ax.get_xlim()[0]),
                #        max(ax.get_ylim()[1], ax.get_xlim()[1])]
                # # ax.set_xlim([-100,100])
                # ax.set_ylim([0, 100])
                x += 1
                legend = False
    y += 1
# fig.tight_layout()
plt.show()
fig.savefig(os.path.join(output_dir, 'training_statistics_maize_resnet18.png'))
# fig.savefig(os.path.join(output_dir, 'training_statistics_soy_resnet18.png'))
# fig.savefig(os.path.join(output_dir, 'training_statistics_sunflowers_resnet18.png'))

# fig.savefig(os.path.join(output_dir, 'training_statistics_maize_baselinemodel.png'))
# fig.savefig(os.path.join(output_dir, 'training_statistics_soy_baselinemodel.png'))
# fig.savefig(os.path.join(output_dir, 'training_statistics_sunflowers_baselinemodel.png'))
