import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
# from ete3 import NCBITaxa
import pickle
import os.path
from os import path
from collections import Counter
import math
from os import listdir
from os.path import isfile, join
import geopandas as gp
import seaborn

def take(elem):
    return elem[1]

def plot_yield_distribution_bar(data_path, figure_path):
    print('plot yield all')
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # ctype = onlyfiles[0].split("_")[-1].split(".")[0]

    df=[]
    ctype = []
    patch_id = []
    for i in range(30):
        csv_path = os.path.join(data_path, onlyfiles[i])
        print(csv_path)
        df.append(pd.read_csv(csv_path))
        ctype.append(onlyfiles[i].split("_")[-1].split(".")[0])
        patch_id.append(onlyfiles[i].split("_")[-2].split(".")[0])

    # yield avg + var all
    pool = []
    avg = []
    sd = []
    for i in range(30):
        yield_avg = 0
        yield_sd = 0
        try:
            pool.extend(df[i]['ertrag'].tolist())
            yield_avg = np.mean(df[i]['ertrag'].tolist())
            yield_sd = np.std(df[i]['ertrag'].tolist())
        except KeyError:
            try:
                pool.extend(df[i]['ERTRAG'].tolist())
                yield_avg = np.mean(df[i]['ERTRAG'].tolist())
                yield_sd = np.std(df[i]['ERTRAG'].tolist())
            except KeyError:
                try:
                    pool.extend(df[i]['Trockenert'].tolist())
                    yield_avg = np.mean(df[i]['Trockenert'].tolist())
                    yield_sd = np.std(df[i]['Trockenert'].tolist())
                except KeyError:
                    print('crap cab')
        avg.append(yield_avg)
        sd.append(yield_sd)
        print('Patch ID: {}; Crop Type: {}; Yield avg: {}; Yield sd: {}'.format(patch_id[i], ctype[i], yield_avg, yield_sd))

    taxonomic_groups = ['Lup', 'Pha', 'Sun', 'SOats', 'Soy', 'Maiz']
    taxonomic_groups_to_color = { 'Lup': 0.857142857142857, 'Pha': 0.714285714285714, 'Sun': 0.571428571428571, 'SOats': 0.428571428571429,
                                 'Soy':  0.285714285714286, 'Maiz': 0.142857142857143}
    cc = [taxonomic_groups_to_color[x] for x in ctype]
    # colormap = {'Lup': 'black', 'Pha': 'red', 'Sun' : 'blue', 'SOats': 'green', 'Soy': 'yellow', 'Maiz': 'purple'}
    # colors = [colormap[x] for x in ctype]
    print(ctype)
    # #receive color map
    # cmap = plt.cm.get_cmap('Dark2', 6)
    cmap = plt.cm.get_cmap('tab10', 6)
    # #encode cc in cmap
    colors = cmap(cc)
    # colors = ['black', 'red', 'yellow', 'black', 'blue', 'orange']

    fig, ax = plt.subplots()
    ax.hlines(y=np.mean(pool), xmin=-0.5, xmax=29.5)
    plt.plot(30,np.mean(pool),marker='<', color='black')
    plt.text(30.50,np.mean(pool)-2, 'mean', rotation='vertical')
    # plt.plot(1,90,marker='<')
    bars=ax.bar(x=range(30), height=avg,
           yerr=sd,
           align='center',
           capsize=10,
           color=colors)

    # first occurance of crop type in list
    first_crop_idx = [ctype.index(x) for x in list(set(ctype))]
    first_crop_bar = [bars[idx] for idx in first_crop_idx]
    ax.legend(first_crop_bar, set(ctype), loc="upper right")
    # ax.legend(loc="upper right")
    ax.set_ylabel('Yield [t/ha]')
    ax.set_xlabel('Patch ID')
    # ax.set_xticks(range(30))
    # ax.set_xticklabels(labels)
    # ax.set_title('Yield')
    ax.yaxis.grid(True)

    plt.xticks(range(30), patch_id, rotation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path,'yields.png'))
    plt.show()
    # yield avg + var for type
    ctypes = set(ctype)
    # yield avg + var patch

    print('done')
def plot_yield_distribution_box(data_path, figure_path):
    print('plot yield all')
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # ctype = onlyfiles[0].split("_")[-1].split(".")[0]

    #geopackage
    # path = "D:/PatchCROP/"
    path = "/media/stillsen/Elements_SE/PatchCROP"
    gpkg = os.path.join(path, "AIA/GeoPackage/", "PatchCrop.gpkg")
    gdf = gp.read_file(gpkg, layer='patch30_ID_WIN_LOC')

    # units = '[t/ha]'
    units = '[kg/m^2]'
    # scale = 1
    scale = 1 / 10
    patch_map = dict(zip(gdf.Patch_ID, zip(gdf.Crop, gdf.Loc, gdf.Win)))
    # print(patch_map)
    # order patch_map to the wanted sort order
    patch_map = {k: v for k, v in sorted(patch_map.items(), key=take)}
    print('wanted order', patch_map)

    df=[]
    ctype_from_file = []
    patch_id_from_file = []
    for i in range(30):
        csv_path = os.path.join(data_path, onlyfiles[i])
        # print(csv_path)
        df.append(pd.read_csv(csv_path))
        ctype_from_file.append(onlyfiles[i].split("_")[-1].split(".")[0])
        patch_id_from_file.append(onlyfiles[i].split("_")[-2].split(".")[0][3:])
    # convert types of patch_id_from_file from str -> int
    # ordered indeces: 1st by crop type, 2nd by win, 3rd by loc
    idx = [list(map(int,patch_id_from_file)).index(item) for item in list(patch_map.keys())]
    print('reordered patch_id_from_file:', [patch_id_from_file[i] for i in idx])
    ##
    # yield avg + var all
    pool = []
    avg = []
    sd = []
    yield_df = []
    for i in range(30):
        yield_avg = 0
        yield_sd = 0
        try:
            cyield = df[i]['ertrag'].tolist()
        except KeyError:
            try:
                cyield = df[i]['ERTRAG'].tolist()
            except KeyError:
                try:
                    cyield = df[i]['Durchflus2'].tolist()
                except KeyError:
                    print('crap cab')
        cyield = list(np.dot(cyield, scale))
        pool.extend(cyield)
        yield_df.append(cyield)
        avg.append(np.mean(cyield))
        sd.append(np.std(cyield))
        # print('Patch ID: {}; Crop Type: {}; Yield avg: {}; Yield sd: {}'.format(patch_id_from_file[i], ctype_from_file[i], yield_avg, yield_sd))
    # apply order to yield_df
    yield_df = [yield_df[i] for i in idx]
    ctype_from_file = [ctype_from_file[i] for i in idx]
    taxonomic_groups = ['Lup', 'Pha', 'Sun', 'SOats', 'Soy', 'Maiz']
    taxonomic_groups_to_color = { 'Lup': 0.857142857142857, 'Pha': 0.714285714285714, 'Sun': 0.571428571428571, 'SOats': 0.428571428571429,
                                 'Soy':  0.285714285714286, 'Maiz': 0.142857142857143}
    cc = [taxonomic_groups_to_color[x] for x in ctype_from_file]
    # colormap = {'Lup': 'black', 'Pha': 'red', 'Sun' : 'blue', 'SOats': 'green', 'Soy': 'yellow', 'Maiz': 'purple'}
    # colors = [colormap[x] for x in ctype]
    print(ctype_from_file)
    # #receive color map
    # cmap = plt.cm.get_cmap('Dark2', 6)
    cmap = plt.cm.get_cmap('tab10', 6)
    # #encode cc in cmap
    colors = cmap(cc)
    # colors = ['black', 'red', 'yellow', 'black', 'blue', 'orange']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()

    # ax1.hlines(y=np.mean(pool), xmin=-0.5, xmax=30.5, color='k')
    # plt.plot(30.5,np.mean(pool),marker='<', color='black')
    # plt.text(31.5,np.mean(pool)-4, 'mean', rotation='vertical')
    # plt.plot(1,90,marker='<')

    boxes =[]
    for i, y in enumerate(yield_df):
        this_box = ax1.boxplot(y,
                              positions=[i+1],
                              widths=[.5],
                              patch_artist=True,
                              boxprops=dict(facecolor=colors[i], color=colors[i]),
                              capprops=dict(color=colors[i]),
                              whiskerprops=dict(color=colors[i]),
                              flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
                              # medianprops=dict(color=colors[i])
                              )
        # LOC create shaded background for low yield zones
        loc = list(patch_map.values())[i][1]
        if loc > 1:
            plt.axvspan(i+.5, i + 1.5, facecolor='0.2', alpha=0.2)
            plt.text(i+.8, -30*scale, "Lo")
        else:
            plt.text(i+.8, -30*scale, "Hi")
        # plot WIN
        win = list(patch_map.values())[i][2]
        top = this_box['boxes'][0].get_path().vertices[2, 1]
        if win == 1:
            plt.text(i+.8,-35*scale, 'c')
            # plt.axvspan(i+.5, i + 1.5, hatch="." , facecolor='0.2', alpha=0.2)
        elif win ==2:
            plt.text(i + .8, -35*scale, 'rp')
            plt.plot(i + 1.25, top+2*scale, marker="*", color="g")
            # plt.axvspan(i + .5, i + 1.5, hatch="/", facecolor='0.2', alpha=0.2)
        elif win == 3:
            plt.text(i + .8, -35*scale, 'rp')
            plt.text(i + .8, -40*scale, '+')
            plt.text(i + .8, -45*scale, 'fs')
            plt.plot(i + 1.25, top+2*scale, marker="*", color="g")
            plt.plot(i + 1.4, top + 2*scale, marker="*", color="g")
            # plt.axvspan(i + .5, i + 1.5, hatch="x", facecolor='0.2', alpha=0.2)
        # plt.text(i + 1, -17, str(win))
        # plt.plot(i+1, -17, marker='*')

        boxes.append(this_box)
        # for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        #     plt.setp(this_box[item], color=colors[i])
    # first occurance of crop type in list
    first_crop_idx = [ctype_from_file.index(x) for x in list(set(ctype_from_file))]
    # boxes[0]['boxes']
    first_crop_box = [boxes[idx]['boxes'][0] for idx in first_crop_idx]
    ax1.legend(first_crop_box, set(ctype_from_file), loc="upper right")
    # ax1.legend(loc="upper right")
    ax1.set_ylabel('Yield '+units)
    ax1.set_xlabel('Patch ID')
    # ax1.set_xticks(range(30))
    # ax1.set_xticklabels(labels)
    # ax1.set_title('Yield')
    ax1.yaxis.grid(True)
    # ax2.set_ylabel('Yield [t/ha]')
    # ax2.set_xlabel('Patch ID')
    # ax2.set_xticks(range(30))
    # ax2.set_xticklabels(labels)
    # ax1.set_title('Yield')

    plt.xlim((0, 31))
    plt.xticks([x+1 for x in range(30)], patch_id_from_file, rotation='vertical')
    plt.tight_layout()
    # plt.savefig(os.path.join(figure_path,'yields.png'))
    plt.show()
    # yield avg + var for type
    ctypes = set(ctype_from_file)
    # yield avg + var patch

    print('done')

def plot_yield_distribution_violin(data_path, figure_path):
    print('plot yield all')
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # ctype = onlyfiles[0].split("_")[-1].split(".")[0]

    #geopackage
    # path = "D:/PatchCROP/"
    path = "/media/stillsen/Elements_SE/PatchCROP"
    gpkg = os.path.join(path, "AIA/GeoPackage/", "PatchCrop.gpkg")
    gdf = gp.read_file(gpkg, layer='patch30_ID_WIN_LOC')

    # units = '[t/ha]'
    units = '[kg/m^2]'
    # scale = 1
    scale = 1 / 10
    patch_map = dict(zip(gdf.Patch_ID, zip(gdf.Crop, gdf.Loc, gdf.Win)))
    # print(patch_map)
    # order patch_map to the wanted sort order
    patch_map = {k: v for k, v in sorted(patch_map.items(), key=take)}
    print('wanted order', patch_map)

    df=[]
    ctype_from_file = []
    patch_id_from_file = []
    for i in range(30):
        csv_path = os.path.join(data_path, onlyfiles[i])
        # print(csv_path)
        df.append(pd.read_csv(csv_path))
        ctype_from_file.append(onlyfiles[i].split("_")[-1].split(".")[0])
        patch_id_from_file.append(onlyfiles[i].split("_")[-2].split(".")[0][3:])
    # convert types of patch_id_from_file from str -> int
    # ordered indeces: 1st by crop type, 2nd by win, 3rd by loc
    idx = [list(map(int,patch_id_from_file)).index(item) for item in list(patch_map.keys())]
    print('reordered patch_id_from_file:', [patch_id_from_file[i] for i in idx])
    ##
    # yield avg + var all
    pool = []
    avg = []
    sd = []
    yield_df = []
    for i in range(30):
        yield_avg = 0
        yield_sd = 0
        try:
            cyield = df[i]['ertrag'].tolist()
        except KeyError:
            try:
                cyield = df[i]['ERTRAG'].tolist()
            except KeyError:
                try:
                    cyield = df[i]['Durchflus2'].tolist()
                except KeyError:
                    print('crap cab')
        cyield = list(np.dot(cyield, scale))
        pool.extend(cyield)
        yield_df.append(cyield)
        avg.append(np.mean(cyield))
        sd.append(np.std(cyield))
        # print('Patch ID: {}; Crop Type: {}; Yield avg: {}; Yield sd: {}'.format(patch_id_from_file[i], ctype_from_file[i], yield_avg, yield_sd))
    # apply order to yield_df
    yield_df = [yield_df[i] for i in idx]
    ctype_from_file = [ctype_from_file[i] for i in idx]
    taxonomic_groups = ['Lup', 'Pha', 'Sun', 'SOats', 'Soy', 'Maiz']
    taxonomic_groups_to_color = { 'Lup': 0.857142857142857, 'Pha': 0.714285714285714, 'Sun': 0.571428571428571, 'SOats': 0.428571428571429,
                                 'Soy':  0.285714285714286, 'Maiz': 0.142857142857143}
    cc = [taxonomic_groups_to_color[x] for x in ctype_from_file]
    # colormap = {'Lup': 'black', 'Pha': 'red', 'Sun' : 'blue', 'SOats': 'green', 'Soy': 'yellow', 'Maiz': 'purple'}
    # colors = [colormap[x] for x in ctype]
    print(ctype_from_file)
    # #receive color map
    # cmap = plt.cm.get_cmap('Dark2', 6)
    cmap = plt.cm.get_cmap('tab10', 6)
    # #encode cc in cmap
    colors = cmap(cc)
    # colors = ['black', 'red', 'yellow', 'black', 'blue', 'orange']

    fig = plt.figure()
    fig.add_subplot(2,3,1)
    # ax2 = ax1.twiny()

    # ax1.hlines(y=np.mean(pool), xmin=-0.5, xmax=30.5, color='k')
    # plt.plot(30.5,np.mean(pool),marker='<', color='black')
    # plt.text(31.5,np.mean(pool)-4, 'mean', rotation='vertical')
    # plt.plot(1,90,marker='<')

    # ax1.violinplot(yield_df)

    # "violine"
    boxes =[]
    j = 0
    prev_crop = ''
    crop_map = list(patch_map.keys())
    for crop in set(crop_map):
        # extract subset
        yields =

        if crop_map[i] != prev_crop:
            j += 1
            prev_crop = crop_map[i]
            ax = fig.add_subplot(2,3,j)
        print((crop,i))
        this_box = ax.violinplot(crop, [i+1], widths=0.3, showmeans=True, showextrema=True, showmedians=True)#, color=colors[j]
        # this_box = ax.boxplot(y,
        #                       positions=[i+1],
        #                       widths=[.5],
        #                       patch_artist=True,
        #                       boxprops=dict(facecolor=colors[i], color=colors[i]),
        #                       capprops=dict(color=colors[i]),
        #                       whiskerprops=dict(color=colors[i]),
        #                       flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
        #                       # medianprops=dict(color=colors[i])
        #                       )
        # LOC create shaded background for low yield zones
        loc = list(patch_map.values())[i][1]
        if loc > 1:
            # plt.axvspan(i+.5, i + 1.5, facecolor='0.2', alpha=0.2)
            plt.text(i+.8, -30*scale, "Lo")
        else:
            plt.text(i+.8, -30*scale, "Hi")
        # plot WIN
        win = list(patch_map.values())[i][2]
        top = this_box['boxes'][0].get_path().vertices[2, 1]
        if win == 1:
            # plt.axvspan(i + .5, i + 1.5, facecolor='0.2', alpha=0.2)
            plt.text(i+.8,-35*scale, 'c')
            # plt.axvspan(i+.5, i + 1.5, hatch="." , facecolor='0.2', alpha=0.2)
        elif win ==2:
            plt.axvspan(i + .5, i + 1.5, facecolor='0.2', alpha=0.2)
            plt.text(i + .8, -35*scale, 'rp')
            # plt.plot(i + 1.25, top+2*scale, marker="*", color="g")
            # plt.axvspan(i + .5, i + 1.5, hatch="/", facecolor='0.2', alpha=0.2)
        elif win == 3:
            plt.axvspan(i + .5, i + 1.5, facecolor='0.2', alpha=0.6)
            plt.text(i + .8, -35*scale, 'rp')
            plt.text(i + .8, -40*scale, '+')
            plt.text(i + .8, -45*scale, 'fs')
            # plt.plot(i + 1.25, top+2*scale, marker="*", color="g")
            # plt.plot(i + 1.4, top + 2*scale, marker="*", color="g")
            # plt.axvspan(i + .5, i + 1.5, hatch="x", facecolor='0.2', alpha=0.2)
        # plt.text(i + 1, -17, str(win))
        # plt.plot(i+1, -17, marker='*')
        ax.set_ylabel('Yield ' + units)
        ax.set_xlabel('Patch ID')
        # ax1.set_xticks(range(30))
        # ax1.set_xticklabels(labels)
        # ax1.set_title('Yield')
        ax.yaxis.grid(True)

        boxes.append(this_box)
        # for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        #     plt.setp(this_box[item], color=colors[i])


    # first occurance of crop type in list
    # first_crop_idx = [ctype_from_file.index(x) for x in list(set(ctype_from_file))]
    # boxes[0]['boxes']
    # first_crop_box = [boxes[idx]['boxes'][0] for idx in first_crop_idx]
    # ax1.legend(first_crop_box, set(ctype_from_file), loc="upper right")
    # ax1.legend(loc="upper right")
    # ax1.set_ylabel('Yield '+units)
    # ax1.set_xlabel('Patch ID')
    # # ax1.set_xticks(range(30))
    # # ax1.set_xticklabels(labels)
    # # ax1.set_title('Yield')
    # ax1.yaxis.grid(True)
    # ax2.set_ylabel('Yield [t/ha]')
    # ax2.set_xlabel('Patch ID')
    # ax2.set_xticks(range(30))
    # ax2.set_xticklabels(labels)
    # ax1.set_title('Yield')

    # plt.xlim((0, 31))
    # plt.xticks([x+1 for x in range(30)], patch_id_from_file, rotation='vertical')
    # plt.tight_layout()
    # # plt.savefig(os.path.join(figure_path,'yields.png'))
    plt.show()
    # yield avg + var for type
    ctypes = set(ctype_from_file)
    # yield avg + var patch

    print('done')

data_path='/home/stillsen/Documents/Zalf/Cleaned_Yield-Maps/yields'
figure_path='/home/stillsen/Documents/GeoData/PatchCROP/AIA/Figures'
# data_path='D:/PatchCROP/AIA/Cleaned_Yield-Maps/yields'
# figure_path='D:/PatchCROP/AIA/Figures'
# plot_yield_distribution_bar(data_path=data_path,figure_path=figure_path)
# plot_yield_distribution_box(data_path=data_path,figure_path=figure_path)
plot_yield_distribution_violin(data_path=data_path,figure_path=figure_path)
