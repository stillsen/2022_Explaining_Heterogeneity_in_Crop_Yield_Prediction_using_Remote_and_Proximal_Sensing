import geopandas as gpd
import shapely
import pandas as pd
from shapely.geometry import Point
from scipy import stats
import os
from matplotlib import pyplot as plt
from shapely.geometry import LineString
from itertools import combinations
import math
def split_line_at_direction_changes(line, epsilon=0.5):
    # Initialize a list to store the split points
    split_points = []

    # Iterate over the coordinates of the line
    for i in range(1, len(line.coords)):
        # Get the current and previous coordinates
        current_coord = line.coords[i]
        previous_coord = line.coords[i - 1]

        # Calculate the direction of the line segment
        dx = current_coord[0] - previous_coord[0]
        dy = current_coord[1] - previous_coord[1]
        direction = (dx, dy)

        # Check if the direction changes
        if i > 1:
            prev_dx = previous_coord[0] - line.coords[i - 2][0]
            prev_dy = previous_coord[1] - line.coords[i - 2][1]
            prev_direction = (prev_dx, prev_dy)

            scalar_prod = dx * prev_dx + dy * prev_dy
            norm = math.sqrt(prev_dx**2 + prev_dy**2) * math.sqrt(dx**2 + dy**2)

            if norm > 0:
                # scalar_product(v1, v2) / (length(v1) * length(v2)) > 1 - epsilon
                if scalar_prod / norm < 1 - epsilon: # i.e. for epsilon 0.5 we allow max 60' angle for directional change
                    # Add the current point as a split point
                    split_points.append(i - 1)
    # Split the line at the split points
    segments = []
    start_index = 0
    for split_index in split_points:
        segment = line.coords[start_index:split_index + 1]
        if len(segment) > 2: # no two piont segments
            segments.append(LineString(segment))
            start_index = split_index

    # Add the last segment
    last_segment = line.coords[start_index:]
    segments.append(LineString(last_segment))

    return segments

def remove_parallel_and_intersecting_paths(gdf, harvest_width, patch_ID, yield_path):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    # multi_line = LineString(gdf.geometry.tolist())
    multi_line = LineString(gdf.geometry.explode().tolist())
    multi_line_df = gpd.GeoSeries([multi_line])
    multi_line_df.plot(ax=axes)
    # df.plot(ax=axes, facecolor='red')
    # df_overlap_removal.plot(ax=axes, facecolor='green')

    # split harvest path into segments for directional changes >= 60 degree angle
    segments = split_line_at_direction_changes(multi_line)


    gdf_seg = gdf
    positive_pdf = gdf
    # positive_pdf.plot(ax=axes, facecolor='red')
    # # loop over segments in harvest order and discard those points that
    for i, segment in enumerate(segments):
        print(segment)
        segment_df = gpd.GeoSeries([segment])
        # segment_df.plot(ax=axes, facecolor='red')

        # buffer for exclusion
        neg_seg_buffer = segment.buffer(harvest_width)
        gpd.GeoSeries([neg_seg_buffer]).plot(ax=axes, facecolor='grey')

        # get yield points that belong to this segment/harvest line
        pos_seg_buffer = segment.buffer(0.2)
        # gpd.GeoSeries([pos_seg_buffer]).plot(ax=axes, facecolor='y')
        gdf_pos_seg = gdf_seg[gdf_seg.geometry.within(pos_seg_buffer)]
        gdf_pos_seg.plot(ax=axes, facecolor='red')

        #exclude yield points of subsequent segments (i:end) within boundaries of segment and not in pos_seg_buffer/on pos seg line
        for j, subsequent_segment in enumerate(segments[i+1:]):
            # compute overlap between segment and subsequent segment
            sub_pos_seg_buffer = subsequent_segment.buffer(0.2)
            gpd.GeoSeries([sub_pos_seg_buffer]).plot(ax=axes, facecolor='grey')
            overlap = neg_seg_buffer.intersection(subsequent_segment)
            # gpd.GeoSeries([overlap]).plot(ax=axes, facecolor='black')
            positive_pdf = positive_pdf[~positive_pdf.geometry.within(overlap)]
    positive_pdf.plot(ax=axes, facecolor='y')
    fig.savefig(os.path.join(yield_path, 'Patch_' + str(patch_ID) + '_filter_parallel-and-intersecting-subsequent-harvest-paths_debug-image.png'))
    # plt.show()

    return positive_pdf

def clean_yield_map(yield_shapefile, patch_shapefile, prefix, patch_ID, path, theta=2):
    # Load yield data from shapefile
    df = gpd.read_file(os.path.join(path, yield_shapefile))
    # df = df.set_crs("EPSG:25833")
    df = df.to_crs("EPSG:25833")
    try:
        df['ertrag'] = df['ertrag']
        df = df.rename({'ertrag': 'yield', 'kornfeu': 'humidity', 'arbbreite': 'width', 'geschwind': 'speed', 'ernt_leist': 'flow'}, axis='columns')
    except KeyError:
        try:
            df['ERTRAG'] = df['ERTRAG']
            df = df.rename({'ERTRAG': 'yield', 'KORNFEU': 'humidity', 'TIMESTAMP': 'timestamp', 'ARBBREITE': 'width', 'GESCHWIND': 'speed', 'ERNT_LEIST': 'flow'}, axis='columns')
        except KeyError:
            df['Ertrag_freu'] = df['Ertrag_feu']
            df = df.rename({'Ertrag_feu': 'yield', 'Feuchtigke': 'humidity', 'Timestamp': 'timestamp', 'Arbeitbre': 'width', 'Geschwindi': 'speed', 'Durchflus2': 'flow'}, axis='columns')

    orig_df = df
    # Load patch data from shapefile
    patch_shapes_all = gpd.read_file(patch_shapefile)
    patch_shapes_all.to_crs("EPSG:25833")

    # Calculate half the harvest width
    harvest_width = (df['width'].mean() / 3)/100

    # Filter points based on proximity to field edges of this patch
    this_patch_shapes = patch_shapes_all[patch_shapes_all['Patch_id_1'] == int(patch_ID)]
    boundary_df = this_patch_shapes.geometry.boundary.buffer(harvest_width)

    # Save buffer as a new shapefile
    # boundary_df.to_file(os.path.join(yield_path, prefix+str(patch_ID)+'_boundary_buffer.shp'))
    boundary_poly = list(boundary_df)[0]
    # discard yield points that harvest trajectory intersects with the field edge
    df_intersect = df[~df.geometry.within(boundary_poly)]
    # discard yield points that lie outside the field
    df_outside = gpd.sjoin(df_intersect, this_patch_shapes, op='within')

    df = df_outside

    # Filter points based on flow (tenth percentile)
    flow_threshold = df['flow'].quantile(0.2)
    df = df[df['flow'] >= flow_threshold]

    # Filter points based on flow (tenth percentile)
    # yield_threshold = df['yield'].quantile(0.05)
    df = df[df['yield'] >= 0.1]

    # Calculate change in speed
    df['speed_change'] = df['speed'].diff()

    # Filter points based on speed change
    df = df[abs(df['speed_change']) <= theta]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    boundary_df.plot(ax=axes)
    orig_df.plot(ax=axes, facecolor='red')
    df.plot(ax=axes, facecolor='green')
    fig.savefig(os.path.join(yield_path, 'Patch_' + str(patch_ID)+'_filter_field-edge_buffer-field-edge_flow_speed_debug-image.png'))

    harvest_width = (df['width'].mean() / 2) / 100
    df = remove_parallel_and_intersecting_paths(df, harvest_width, patch_ID, yield_path)

    # plt.show()

    return df

# yield_path = '../../1_Data_working_copy/2021_Yield Maps_Raw/Yield_Maps_All'
yield_path = '/media/stillsen/Hinkebein/PatchCROP/AIA/2022_Explaining_Heterogeneity_in_Crop_Yield_Prediction_using_Remote_and_Proximal_Sensing/1_Data_working_copy/2022_YieldMaps_Raw/Yield_Maps_All'
patch_shape_path = '../../1_Data_working_copy/Shapes'

# Example usage
# yield_shapefile = os.path.join(yield_path, 'Patch_81_WG_16.07.2021 13_58_36_WGS84.shp')
# yield_shapefile = os.path.join(yield_path, 'Patch_65_KM_10.11.2021 14_34_47_WGS84.shp')
# yield_shapefile = os.path.join(yield_path, 'Patch_39_RA_23.07.2021 13_32_01_WGS84.shp')
patch_shapefile = os.path.join(patch_shape_path,'patch30_min_flowerstrips.shp')
# patch_ID = 81
# patch_ID = 65
# patch_ID = 39

for file in os.listdir(yield_path):
    if file.endswith(".shp"):
        print(os.path.join(yield_path, file))
        patch_ID = file.split('_')[1]
        prefix = 'Patch_{}__cleaned'.format(patch_ID)
        cleaned_yield_data = clean_yield_map(file, patch_shapefile, prefix, patch_ID, yield_path)
        # # Save the cleaned yield data to a new shapefile
        cleaned_yield_data.to_file(os.path.join(yield_path, prefix+'.shp'))
        #
        # # Save the cleaned yield data to a CSV file
        # cleaned_yield_data.to_csv(prefix+'.csv', index=False)