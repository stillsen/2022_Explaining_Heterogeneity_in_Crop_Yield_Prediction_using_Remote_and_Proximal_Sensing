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
import matplotlib.offsetbox as osb
from matplotlib.colors import Normalize

# for resizing images to thumbnails
import torchvision.transforms.functional as functional
from matplotlib import rcParams as rcp
from PIL import Image

from umap import UMAP

# for clustering and 2d representations
from sklearn import random_projection
# Own modules
from PatchCROPDataModule import PatchCROPDataModule#, KFoldLoop#, SpatialCVModel
from directory_listing import output_dirs, data_dirs, input_files_rgb


__author__ = 'Stefan Stiller'
__copyright__ = 'Copyright 2022, Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing'
__credits__ = ['Stefan Stiller, Gohar Ghazaryan, Kathrin Grahmann, Masahiro Ryo']
__license__ = 'GNU GPLv3'
__version__ = '0.1'
__maintainer__ = 'Stefan Stiller'
__email__ = 'stefan.stiller@zalf.de, stillsen@gmail.com'
__status__ = 'Dev'



def get_scatter_plot_with_thumbnails(embeddings_2d, img_stack, color, fig, kernel_size):
    """Creates a scatter plot with image overlays."""
    # initialize empty figure and add subplot
    if fig == None:
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle("Scatter Plot of Maize for {}x{}".format(kernel_size,kernel_size))
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.gca()
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        # if np.min(dist) < 2e-3:
        # if np.min(dist) < 2e-2:
        if np.min(dist) < 1.9:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # ax.scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)
    ax.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], marker='.', c=color, alpha=0.8)
    yields = [img_stack[i][1] for i in range(len(img_stack))]
    min_yield = np.min([img_stack[i][1] for i in range(len(img_stack))])
    max_yield = np.max([img_stack[i][1] for i in range(len(img_stack))])
    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 8.0)
        img = img_stack[idx][0]
        img = functional.resize(img, thumbnail_size)
        img = img.squeeze().movedim(0, 2).numpy()

        yield_value = img_stack[idx][1]
        # color = plt.cm.viridis((yield_value - min_yield) / (max_yield - min_yield))
        color = plt.cm.RdYlGn((yield_value - min_yield) / (max_yield - min_yield))

        img_box = osb.AnnotationBbox(
            osb.OffsetImage(img, cmap=plt.cm.gray_r),
            embeddings_2d[idx],
            pad=0.2,
            frameon=True,
            boxcoords="data",
            bboxprops=dict(edgecolor=color, linewidth=2)
        )
        ax.add_artist(img_box)

    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    # ratio = 1.0 / ax.get_data_ratio() *0.05
    ax.set_aspect(ratio, adjustable="box")

    # Add color bar
    norm = Normalize(vmin=min_yield, vmax=max_yield)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])  # dummy empty array for the colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Crop Yield')

    return fig



# output_dirs = dict()
# data_dirs = dict()
# input_files = dict()
# input_files_rgb = dict()

# data_root = '/beegfs/stiller/PatchCROP_all/Data/'
data_root = '../../2_Data_preprocessed/2977x_Raster_Rescaled_Labels_and_Features__Analyses_Packages_for_HPC/'
output_root = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/Output/SSL/Tests/'
# output_root = '/beegfs/stiller/PatchCROP_all/Output/L2/SSL/Tests/'

# output_dirs[65] = os.path.join(output_root, 'P_65_resnet18_SCV_no_test_L2_cycle_E1000')
# output_dirs[68] = os.path.join(output_root, 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224-black-added-to-train-and-val')
# output_dirs[68] = os.path.join(output_root, 'P_68_VICRegLConvNext_SCV_no_test_E100lightly-VICRegLConvNext_kernel_size_224_recompute')
output_dirs[68] = os.path.join(output_root, 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224_recompute')
# output_dirs[68] = os.path.join(output_root, 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224-black-added-to-train')
# output_dirs['combined_train'] = os.path.join(output_root, 'Combined_crops_train_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224')
# output_dirs['combined_test'] = os.path.join(output_root, 'Combined_crops_train_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224')
# output_dirs[68] = os.path.join(output_root, 'P_68_VICReg_SCV_no_test_E100lightly-VICReg_kernel_size_224')
# output_dirs[76] = os.path.join(output_root, 'P_76_baselinemodel_RCV_L2_cycle_E1000')

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
# patch_no = 'combined_train'
# patch_no = 'combined_test'
# test_patch_no = 90

num_folds = 4

this_output_dir = output_dirs[patch_no]

# test_patch_name = 'Patch_ID_'+str(patch_no)+'_grn'
# test_patch_name = 'Patch_ID_'+str(patch_no)
test_patch_name = ''

seed = 42
stride = 30
kernel_size = 224
# add_n_black_noise_img_to_train = 1500
# add_n_black_noise_img_to_val = 500
augmentation = False
features = 'RGB'
# features = 'black'
batch_size = 1
# validation_strategy = 'RCV'
validation_strategy = 'SCV_no_test'  # SHOV => Spatial Hold Out Validation; SCV => Spatial Cross Validation; SCV_no_test; RCV => Random Cross Validation
num_samples_per_fold = None
fake_labels = False

# Detect if we have a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('working on device %s' % device)
if device == 'cpu':
    workers = os.cpu_count()
else:
    workers = 1#torch.cuda.device_count()
    print('\twith {} workers'.format(workers))

datamodule = PatchCROPDataModule(
                                 input_files=input_files_rgb[patch_no],
                                 # input_files=input_files_rgb[directory_id],
                                 patch_id=patch_no,
                                 this_output_dir=this_output_dir,
                                 seed=seed,
                                 data_dir=data_dirs[patch_no],
                                 # data_dir=data_dirs[directory_id],
                                 stride=stride,
                                 kernel_size=kernel_size,
                                 workers=workers,
                                 augmented=augmentation,
                                 input_features=features,
                                 batch_size=batch_size,
                                 validation_strategy=validation_strategy,
                                 fake_labels=fake_labels,
                                 )
datamodule.prepare_data(num_samples=num_samples_per_fold)

sets = [
        '_train_',
        '_val_',
        # '_test_',
        ]

for s in sets:
### Combined Plots - all folds in one figure

    # fig = None
    for k in range(num_folds):
    # for k in range(1):
    #     if k==1: continue
        fig = None
        print('combined plots: {}th-fold'.format(k))
        # load y and y_hat
        y = torch.load(os.path.join(this_output_dir, test_patch_name+'y_SSL' + s + str(k) + '.pt'))
        embeddings = torch.load(os.path.join(this_output_dir, test_patch_name+'y_hat_SSL' + s + str(k) + '.pt'))

        # for the scatter plot we want to transform the images to a two-dimensional
        # vector space using a random Gaussian projection
        # projection = random_projection.GaussianRandomProjection(n_components=2, random_state=seed, eps=0.1)
        # embeddings_2d = projection.fit_transform(embeddings)

        # umap_2d = UMAP(n_components=2, init='random', random_state=seed)
        umap_2d = UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=25,
            min_dist=0.1,
            # metric=metric
        )
        embeddings_2d = umap_2d.fit_transform(embeddings)

        # # normalize the embeddings to fit in the [0, 1] square
        # M = np.max(embeddings_2d, axis=0)
        # m = np.min(embeddings_2d, axis=0)
        # embeddings_2d = (embeddings_2d - m) / (M - m)

        # plot in combined figure
        # ax.scatter(x=y_hat, y=y, marker='.', c=colors[k], alpha=0.8)

        datamodule.setup_fold(
            fold=k,
            training_response_standardization=False,
            # add_n_black_noise_img_to_train=add_n_black_noise_img_to_train,
            # add_n_black_noise_img_to_val=add_n_black_noise_img_to_val,
        )
        if s == '_train_' : image_set = datamodule.train_fold
        elif s == '_val_' : image_set = datamodule.val_fold
        elif s == '_test_': image_set = datamodule.test_fold
        # get a scatter plot with thumbnail overlays
        fig = get_scatter_plot_with_thumbnails(embeddings_2d=embeddings_2d,img_stack=image_set, color=colors[k], fig=fig, kernel_size=kernel_size)
        fig.tight_layout()
        fig.savefig(os.path.join(this_output_dir, test_patch_name + 'embedding_scatterplot{}{}'.format(s[:-1], k) +'.png'), dpi=300)
    # plt.show()
    print()



