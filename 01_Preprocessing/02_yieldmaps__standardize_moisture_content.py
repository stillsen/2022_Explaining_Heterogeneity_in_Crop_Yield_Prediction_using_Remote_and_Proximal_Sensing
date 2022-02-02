import os.path

import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from geopandas import GeoDataFrame, read_file
import seaborn as sns

import matplotlib.pyplot as plt

path = "/media/stillsen/Elements_SE/PatchCROP/AIA/GeoPackage/Cleaned_Yield-Maps_AIA_Working_copy/flow_corrected_manually_corrected**"
smc = {'Maiz': 0.155, 'Pha': 0.12, 'Sun': 0.1, 'Lup': 0.15, 'SOats': 0.14, 'Soy': 0.13}

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
        print("This dataset contains {} records.".format(len(df)))
        crop = file.split('_')[-1].split('.')[0]
        ones = pd.Series(np.repeat([1],len(df)))
        smc_vec = pd.Series(np.repeat(smc[crop],len(df)))
        # df.assign(yield_smc= df['yield']*(ones-df['humidity']/100)/(ones-smc_vec))
        df['yield_smc'] = df['yield'] * (ones - df['humidity'] / 100) / (ones - smc_vec)
        # print(df['yield'].values)
        # print(df['yield_smc'].values)
        df.to_file(os.path.join(path, 'standard_moisture_content', file))

