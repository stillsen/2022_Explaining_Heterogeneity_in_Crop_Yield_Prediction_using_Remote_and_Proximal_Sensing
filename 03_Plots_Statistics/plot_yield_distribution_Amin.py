import glob
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt

# fontsize = 9
fontsize = 12
def plot_PC_patch_yield(lst_patch_id, data_path):
    final_df = pd.DataFrame()
    # for shp_file in glob.glob(data_path + "/*.shp"):
    for shp_file in glob.glob(data_path + "/*.shp")[::-1]:
        for Id in lst_patch_id:
            if Id in shp_file:
                shape_data = gpd.read_file(shp_file)
                final_df[Id] = shape_data.yield_smc

    colors_list = ['#78C850', '#F08030', '#6890F0', '#A8B820', '#F8D030',
                   '#E0C068', '#C03028', '#F85888', '#98D8D8']
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=final_df, linewidth=1, palette=colors_list)
    sns.swarmplot(data=final_df, color="k", alpha=0.8)
    # plt.title(" Yield Distribution - Maize", fontsize=fontsize, fontweight='bold')
    # plt.title(" Yield Distribution - Soy", fontsize=fontsize, fontweight='bold')
    # plt.title(" Yield Distribution - Sunflowers", fontsize=fontsize, fontweight='bold')
    plt.ylabel("Yield [dt/ha]", fontsize=fontsize, fontweight='bold')
    plt.xlabel("Field ID", fontsize=fontsize, fontweight='bold')
    plt.grid()
    plt.tight_layout()
    # plt.savefig("distribution_plot_Maize_P_68_90.png", dpi=400)
    # plt.savefig("distribution_plot_Soy_P_19_65.png", dpi=400)
    plt.savefig("distribution_plot_Sunflowers_P_76_95.png", dpi=400)


if __name__ == "__main__":
    # data_root = "/Users/amin/Desktop/geopandas_project_Zalf/shape_files"
    data_root = "/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/1_Data_working_copy/Cleaned_Yield-Maps_AIA_Working_copy/flow_corrected_manually_corrected/standard_moisture_content"
    # lst_id = ['68','90']#["20", "50", "68", "74", "90", "110" ]
    # lst_id = ["19","65"]# "19", "58", "65" ]
    lst_id = ['76','95']#["76", "95", "115" ]
    plot_PC_patch_yield(lst_id, data_root)