import geopandas as gpd
import pandas as pd
import sqlite3

def prepare_taxi(db_path, grids_path, taxi_joined_path, taxi_ts_path, threshold = 50):
    ## 将车牌和格网 spatial join
    grids = gpd.read_file(grids_path)

    conn = sqlite3.connect(db_path)
    ods = pd.read_sql("select * from TAXI", conn)
    conn.close()
    print(ods.shape)

    del ods["ID"]
    ods_geo = gpd.GeoDataFrame(ods, geometry=gpd.points_from_xy(ods["lon"], ods["lat"]), crs=4326)
    ods_geo = ods_geo.to_crs(grids.crs)

    print("spatial join started")
    res = gpd.sjoin(ods_geo, grids, how="inner")
    print("spatial join end")

    res = res[["timestamp", "OD", "GID", "lon", "lat"]]
    res.to_csv(taxi_joined_path)

    ## 处理为时间序列
    data = res

    # 筛选出有一定数据量的格网

    tmp_s = data.groupby(["GID"]).size()
    valid_tmp_s = tmp_s[tmp_s > threshold]
    print(valid_tmp_s.shape[0])
    valid_data = data[data["GID"].isin(list(valid_tmp_s.index))]

    valid_data["hour"] = valid_data["timestamp"].str[11:13]

    # 计OD的checkin统计
    valid_time_od_data = valid_data.groupby(["GID", "hour", "OD"]).size().unstack("OD").unstack("hour")
    valid_time_od_data = valid_time_od_data.fillna(0)

    # 多重列索引合并为一个索引
    valid_time_od_data.columns = ['_'.join(col).strip() for col in valid_time_od_data.columns.values]

    valid_time_od_data.to_csv(taxi_ts_path)