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
grid_size = 500
boundary_path = "./data/raw/sz_qu.json"
ground_truth_path = "./data/raw/urbanVillage_sz/urbanVillage_sz_landtruth_2018.shp"
rs_tiff_path = "./data/raw/shenzhen.tiff"
taxi_db_path = "./data/taxi_valid2.db"
sv_xy_info_path = "./data/raw/SV/shenzhen_all_50m_xy.csv"
sv_path_path = "./data/raw/SV/shenzhen_all_50m_path.csv"

# Outputs
## output path for grids geojson
grids_geojson_path = "./data/grids_all_500m/grids_all_500m.geojson"
## output path for grids labels csv
grids_label_path = "./data/grids_all_500m/grids_label_all_500m.csv"
## output folder for remote sensing tiles
rs_tiles_folder = "./data/grids_all/rs_tiles"
# output path for taxi data
taxi_joined_path = "./data/taxi_all_valid_500m.csv"
taxi_ts_path = "./data/taxi_all_48f.csv"
# output path for street view data
valid_sv_path = "./data/grids_all_500m/sv_info_within_valid.csv"
valid_sv_path_path = "./data/grids_all_500m/sv_path_id_gid.csv"


###################################################
####################  Steps #######################
###################################################

## Step 1: Generate grids
from preprocessing.gen_grids import gen_grids

gen_grids(grid_size, boundary_path, ground_truth_path, grids_geojson_path, grids_label_path)

## Step 2: Prepare remote sensing data - Split tiles from huge GeoTiff file based on grids
from preprocessing.split_geotiff import split_image_by_shapefile
split_image_by_shapefile(rs_tiff_path, grids_geojson_path, rs_tiles_folder)

## Step 3: Prepare taxi data - Joined to grids and calculate the time series
from preprocessing.taxi_process import prepare_taxi
prepare_taxi(taxi_db_path, grids_geojson_path, taxi_joined_path, taxi_ts_path)

## Step 4: Prepare street view data - Joined to grids
from preprocessing.sv_join import prepare_sv
prepare_sv(sv_xy_info_path, sv_path_path, grids_geojson_path, valid_sv_path, valid_sv_path_path)