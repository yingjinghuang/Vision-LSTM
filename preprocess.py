import warnings
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

###################################################
##############  Config part #######################
###################################################
# Parameters
grid_size = 250
valid_ratio = 0.2
boundary_path = r".\data\raw\sz_qu.json"
ground_truth_path = r".\data\raw\urbanVillage_sz\urbanVillage_sz_landtruth_2018.shp"
rs_tiff_path = "./data/raw/shenzhen.tiff"
taxi_db_path = r".\data\taxi_valid2.db"
sv_xy_info_path = r".\data\raw\shenzhen_all_50m_xy.csv"
sv_path_path = r".\data\raw\shenzhen_all_50m_path.csv"

# Outputs
output_folder = f"Vision-LSTM/data/grids{grid_size}"
## output path for grids geojson
grids_geojson_path = os.path.join(output_folder, "grids.geojson")
## output path for grids labels csv
grids_label_path = os.path.join(output_folder, "grids_label.csv")
## output folder for remote sensing tiles
rs_tiles_folder = os.path.join(output_folder, "rs_tiles")
# output path for taxi data
taxi_joined_path = os.path.join(output_folder, "taxi_valid.csv")
taxi_ts_path = os.path.join(output_folder, "taxi_ts340.csv")
# output path for street view data
valid_sv_path = os.path.join(output_folder, "sv_path_id_gid.csv")
valid_feature_path = os.path.join(output_folder, "sv_features.pkl")
# output path for model data
model_data_path = os.path.join(output_folder, "model_data.pkl")
model_data_geo_path = os.path.join(output_folder, "model_data_geo.shp")

###################################################
####################  Steps #######################
###################################################

# Step 1: Generate grids
from preprocessing.gen_grids import gen_grids
gen_grids(grid_size, boundary_path, ground_truth_path, grids_geojson_path, grids_label_path)

## Step 2: Prepare remote sensing data - Split tiles from huge GeoTiff file based on grids
from preprocessing.split_geotiff import split_image_by_shapefile
split_image_by_shapefile(rs_tiff_path, grids_geojson_path, rs_tiles_folder)

# Step 3: Prepare taxi data - Joined to grids and calculate the time series
from preprocessing.taxi_process import prepare_taxi
prepare_taxi(taxi_db_path, grids_geojson_path, taxi_joined_path, taxi_ts_path)

## Step 4: Prepare street view data - Joined to grids
from preprocessing.sv_process import sv_join, features_join
sv_join(sv_xy_info_path, sv_path_path, grids_geojson_path, valid_sv_path)
features_join(valid_sv_path, sv_features_path, valid_feature_path)

## Step 5: Build datasets
from preprocessing.build_dataset import build_dataset
build_dataset(valid_sv_path_path, taxi_ts_path, grids_label_path, valid_ratio, grids_geojson_path, model_data_path, model_data_geo_path)
