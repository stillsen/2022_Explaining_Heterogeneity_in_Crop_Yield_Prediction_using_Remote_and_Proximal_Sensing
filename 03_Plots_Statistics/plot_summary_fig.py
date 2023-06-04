import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# root_dir = '/beegfs/stiller/PatchCROP_all/Output/'
root_dir = '../../Output/shuffle/L2/'

exclude_these_subdirs = [ 'P_68_resnet18_SCV_RCV',
                          'P_65_resnet18_SCV_RCV',
                          'P_76_resnet18_SCV_RCV',
                          'P_68_baselinemodel_SCV_RCV',
                          'P_65_baselinemodel_SCV_RCV',
                          'P_76_baselinemodel_SCV_RCV',
                          'P_68_resnet18_SCV_no_test_SSL_L2_cycle_E1000_resetW'
                          ]


# SSL_suffix = '_domain-tuning'
# SSL_suffix = '_self-supervised'
SSL_suffix = ''
num_folds = 4



font_size = 15
# root_dir = '/path/to/root/directory'
plt.rc('font', size=font_size)          # controls default text sizes
plt.rc('axes', titlesize=font_size-2)     # fontsize of the axes title
plt.rc('axes', labelsize=font_size-2)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size-2)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size-2)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)    # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

fig, axes = plt.subplots(1)

result_df_glob = pd.DataFrame(columns=['r', 'crop_type', 'architecture', 'CV', 'set', 'sort'])
result_df_loc = pd.DataFrame(columns=['r', 'crop_type', 'architecture', 'CV', 'set'])

for subdir in os.listdir(root_dir):
    if subdir in exclude_these_subdirs:
        continue
    if '65' in subdir: crop_type = 'soy'
    if '68' in subdir: crop_type = 'maize'
    if '76' in subdir: crop_type = 'sunflowers'
    if 'baselinemodel' in subdir: architecture = 'base'
    if 'resnet18' in subdir: architecture = 'resnet18'
    if 'RCV' in subdir: cv = 'RCV'
    if 'SCV' in subdir: cv = 'SCV'

    subdir_path = os.path.join(root_dir, subdir)
    print(subdir_path)
    if os.path.isdir(subdir_path):
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            if csv_file.startswith('performance_metrics_f3_') and ('_train_' in csv_file or '_val_' in csv_file): # train and internal val
                csv_path = os.path.join(subdir_path, csv_file)
                csv_df = pd.read_csv(csv_path)
                if ('_train_' in csv_file):
                    if 'RCV' in subdir:
                        sort = 1
                    else:
                        sort = 4
                    result_df_glob.loc[subdir + '_train'] = [csv_df['global_r'].values[0], crop_type, architecture, cv, 'train', sort]
                    result_df_loc.loc[subdir+'_train'] = [csv_df['local_r_median'].values[0], crop_type, architecture, cv, 'train']
                if ('_val_' in csv_file):
                    if 'RCV' in subdir:
                        sort = 2
                    else:
                        sort = 5
                    result_df_glob.loc[subdir+'_val'] = [csv_df['global_r'].values[0], crop_type, architecture, cv, 'val', sort]
                    result_df_loc.loc[subdir+'_val'] = [csv_df['local_r_median'].values[0], crop_type, architecture, cv, 'val']

            if csv_file.startswith('performance_metrics_f3_') and ('_test_' in csv_file): # external val
                csv_path = os.path.join(subdir_path, csv_file)
                csv_df = pd.read_csv(csv_path)
                if 'RCV' in subdir:
                    sort = 3
                else:
                    sort = 6
                result_df_glob.loc[subdir+'_test'] = [csv_df['average_r'].values[0], crop_type, architecture, cv, 'test', sort]
                result_df_loc.loc[subdir+'_test'] = [csv_df['average_r'].values[0], crop_type, architecture, cv, 'test']

result_df_glob.to_csv(os.path.join(root_dir, 'summary_glob.csv'), encoding='utf-8')
result_df_loc.to_csv(os.path.join(root_dir, 'summary_loc.csv'), encoding='utf-8')

agg_df_loc = result_df_loc.groupby(['CV','set'])[['r']].agg(['mean', 'std', 'count'])
agg_df_glob = result_df_glob.groupby(['CV','set'])[['r']].agg(['mean', 'std', 'count'])

grouped_df_glob = result_df_glob.groupby(['CV','set'])

x = [s[0]+' '+s[1] for s in agg_df_glob.index]
y = agg_df_glob[('r','mean')].values
error = agg_df_glob[('r','std')].values
print(x)
print(y)
print(error)
x = [x[i] for i in [1,2,0,4,5,3]]
y = [y[i] for i in [1,2,0,4,5,3]]
error = [error[i] for i in [1,2,0,4,5,3]]
print(x)
print(y)
print(error)
# colors_list = ['#78C850', '#F08030', '#6890F0', '#A8B820', '#F8D030',
#                    '#E0C068', '#C03028', '#F85888', '#98D8D8']
# colors = [colors_list[4],
#          colors_list[4],
#          colors_list[4],
#          colors_list[8],
#          colors_list[8],
#          colors_list[8]]
colors = [
    '#C875C4',
    '#C875C4',
    '#C875C4',
    '#029386',
    '#029386',
    '#029386',
]
alphas = [1.,
          0.8,
          0.6,
          1.,
          0.8,
          0.6,
          ]
for i in range(6):
    axes.errorbar(x[i], y[i], yerr=error[i], fmt='D', markersize='10', color=colors[i], ecolor='grey', lw=2)
# result_df_glob.boxplot(ax=axes, column='r',by=['CV','set'],positions=[2,0,1,5,3,4], showfliers=False,boxprops= dict(linewidth=2.0), whiskerprops=dict(linestyle='-',linewidth=2.0))

plt.grid(visible=True)
axes.grid(axis='x')
axes.set_ylabel('r', fontsize=font_size)
# result_df_glob.boxplot(column='r',by=['CV','set'])
axes.set_title('')
plt.title('')
plt.show()
fig.tight_layout()
fig.savefig(os.path.join(root_dir, 'summary.png'))
fig.savefig(os.path.join(root_dir, 'summary.svg'))
