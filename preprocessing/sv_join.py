import pandas as pd
import geopandas as gpd

def prepare_sv(sv_xy_info_path, sv_path_path, grids_path, valid_sv_path, valid_sv_path_path):

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
    print("Valid grid count：", len(res["GID"].unique()))

    sv_valid = res[["id", "lat_wgs", "lon_wgs", "pov_exp", "heading", "GID"]]
    sv_valid.to_csv(valid_sv_path)

    ## 连接路径
    sv_path = pd.read_csv(sv_path_path, index_col=0)
    sv_path["id"] = sv_path["org_name"].str.split("_", expand=True)[0]
    sv_path["new_path"] = sv_path["new_path"].str.replace("Tencent_SV/city_image/guangdong/shenzhen_all_50m", "data/raw/SV/SZTSV/shenzhen_all_50m")
    sv_path.rename(columns={"new_path": "path"}, inplace=True)
    sv_path = sv_path.merge(sv_valid, on="id")

    sv_path[["id", "path", "GID"]].to_csv(valid_sv_path_path)