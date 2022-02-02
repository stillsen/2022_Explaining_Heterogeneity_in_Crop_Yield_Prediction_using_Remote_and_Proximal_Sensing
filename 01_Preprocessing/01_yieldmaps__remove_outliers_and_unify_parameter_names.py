import os.path

import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from geopandas import GeoDataFrame, read_file
import seaborn as sns

import matplotlib.pyplot as plt

path = "/media/stillsen/Elements_SE/PatchCROP/AIA/GeoPackage/Cleaned_Yield-Maps_AIA_Working_copy**"
# ormo = os.path.join(path, "/media/stillsen/Elements_SE/PatchCROP_MSI/pC_col_plant_drone_Multi_20200716/4_index/reflectance/", "Tempelberg_sequ_16072020_transparent_reflectance_green.tif")
lower_percentile = .2
upper_percentile = 1

for file in os.listdir(path):
    # within a patch
    if file.endswith(".shp"):
        print(os.path.join(path, file))
        # read yield shape file, process it, i.e.
        # 1) geopandas.df and set crc
        # 2) remove lower and upper 10 percentile
        # exception handling is due inconsistent naming
        df = read_file(os.path.join(path, file))
        df = df.set_crs("EPSG:25833")
        try:
            df['ertrag'] = df['ertrag']
            df = df.rename({'ertrag':'yield', 'kornfeu':'humidity', 'arbbreite':'with', 'geschwind':'speed', 'ernt_leist':'flow'}, axis='columns')
        except KeyError:
            try:
                df['ERTRAG'] = df['ERTRAG']
                df = df.rename({'ERTRAG':'yield', 'KORNFEU':'humidity', 'ARBBREITE':'with', 'GESCHWIND':'speed', 'ERNT_LEIST':'flow'}, axis='columns')
            except KeyError:
                df['Ertrag_freu'] = df['Ertrag_feu']
                df = df.rename({'Ertrag_feu':'yield', 'Feuchtigke':'humidity', 'Arbeitbre':'with', 'Geschwindi':'speed', 'Durchflus2':'flow'}, axis='columns')
        qdf = df[df.flow < df.flow.quantile(upper_percentile)]
        qdf = qdf[qdf.flow > df.flow.quantile(lower_percentile)]
        print("This dataset contains {} records.\n {} records are discarded".format(len(df), len(df)-len(qdf)))
        print("upper flow percentile: {}".format(df.flow.quantile(upper_percentile)))
        print("lower flow percentile: {}".format(df.flow.quantile(lower_percentile)))
        qdf.to_file(os.path.join(path, 'flow_corrected', file))

        # sns.distplot(df['flow'], hist=True, kde=True,
        #              bins=int(len(df) / 30), color='darkblue',
        #              hist_kws={'edgecolor': 'black'},
        #              kde_kws={'linewidth': 4})
        plt.figure()
        sns.distplot(df['flow'], hist=True,
                     bins=int(len(df) / 20), color='darkblue',
                     hist_kws={'edgecolor': 'black'})
        plt.axvline(df.flow.quantile(lower_percentile),0,1)
        plt.savefig(os.path.join(path, 'flow_corrected', file[:-3]+'png'))
