import pandas as pd
import geopandas as gpd
import glob, os
import numpy as np

def sv_join(sv_xy_info_path, sv_path_path, grids_path, valid_sv_path):

    xy_info = pd.read_csv(sv_xy_info_path, index_col=0)
    data_geo = gpd.GeoDataFrame(xy_info, geometry=gpd.points_from_xy(xy_info["lon_wgs"], xy_info["lat_wgs"]), crs=4326)
    grids = gpd.read_file(grids_path)
    data_geo = data_geo.to_crs(grids.crs)
    # 求交集
    sv_valid = data_geo[data_geo.intersects(grids.unary_union)]

    # 研究区内的街景点数
    print(sv_valid.shape)

    ## 与格网空间连接
    res = gpd.sjoin(sv_valid, grids[["GID", "geometry"]])
    res = res.reset_index()
    print("Valid grid count：", len(res["GID"].unique()))

    sv_valid = res[["id", "lat_wgs", "lon_wgs", "pov_exp", "heading", "GID"]]

    ## 连接路径
    sv_path = pd.read_csv(sv_path_path, index_col=0)
    sv_path["id"] = sv_path["org_name"].str.split("_", expand=True)[0]
    sv_path["new_path"] = sv_path["new_path"].str.replace("Tencent_SV/city_image/guangdong/shenzhen_all_50m", "data/raw/SV/SZTSV/shenzhen_all_50m")
    sv_path.rename(columns={"new_path": "path"}, inplace=True)
    sv_path = sv_path.merge(sv_valid, on="id")

    sv_path[["id", "path", "GID"]].to_csv(valid_sv_path)

def features_join(valid_sv_path, sv_features_path, valid_feature_path):
    sv = pd.read_csv(valid_sv_path, dtype = {'GID': str}, index_col=0)
    sv["name"] = sv["path"].apply(os.path.basename)

    tmp_list = []
    for file in glob.glob(sv_features_path):
        tmp = pd.read_csv(file, index_col=0)
        del tmp["id"]
        tmp["path"] = tmp["path"].apply(os.path.basename)
        tmp_list.append(tmp)
    features = pd.concat(tmp_list)

    sv = pd.merge(sv, features, how="left", left_on="name", right_on="path")
    data_l = []
    for name, group in sv.groupby("GID"):
        tmp_features = np.array(group[[str(x) for x in list(range(512))]])
        data_l.append([name, tmp_features])

    grid_features = pd.DataFrame(data_l)
    grid_features.columns = ["GID", "features"]
    grid_features.to_pickle(valid_feature_path)
