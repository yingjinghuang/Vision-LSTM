import pandas as pd
import geopandas as gpd
import numpy as np
import os

def build_dataset(valid_feature_path, taxi_ts_path, rs_tiles_folder, grids_label_path, valid_ratio, grids_geojson_path, model_data_path, model_data_geo_path):
    sv = pd.read_pickle(valid_feature_path)
    taxi = pd.read_csv(taxi_ts_path, dtype = {"GID": str}, index_col=0)
    taxi = taxi.reset_index()
    labels = pd.read_csv(grids_label_path, dtype = {'GID': str}, index_col=0)
    intersect_set = list(set(sv["GID"]).intersection(set(taxi["GID"])))
    for GID in intersect_set:
        if not os.path.exists(os.path.join(rs_tiles_folder, GID + ".tif")):
            intersect_set.remove(GID)
    print("Intersect size: ", len(intersect_set))

    sv_valid = sv[sv["GID"].isin(list(intersect_set))]
    taxi_valid = taxi[taxi["GID"].isin(list(intersect_set))]
    labels_valid = labels[labels["GID"].isin(list(intersect_set))]

    data = pd.merge(sv_valid, taxi_valid, on = "GID")
    data = pd.merge(data, labels_valid[["GID", "label"]], on = "GID")

    val_index = data.sample(frac=valid_ratio).index.tolist()
    data["mode"] = "train"
    data.loc[val_index, "mode"] = "val"
    print(data[["label", "mode"]].value_counts())
    data.to_pickle(model_data_path)

    grids = gpd.read_file(grids_geojson_path)
    grids_valid = grids[grids["GID"].isin(list(intersect_set))]
    grids_valid = pd.merge(grids_valid, data[["GID", "mode"]])
    grids_valid = gpd.GeoDataFrame(grids_valid, geometry="geometry")
    grids_valid.to_file(model_data_geo_path)