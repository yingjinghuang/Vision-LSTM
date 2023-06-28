import geopandas as gpd
import pandas as pd
import numpy as np
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
    res["hour"] = res["timestamp"].str[11:13]
    res["date"] = res["timestamp"].str[:10]

    taxi_time_od_data = res.groupby(["GID", "date", "hour", "OD"]).size().unstack("OD").unstack("date").unstack("hour")
    taxi_time_od_data = taxi_time_od_data.fillna(0)
    taxi_time_od_data.columns = ['_'.join(col).strip() for col in taxi_time_od_data.columns.values]
    # 数据仅从 2017-10-22 23点 至 2017-10-30 00点
    col_name = list(taxi_time_od_data.columns[23:193]) + list(taxi_time_od_data.columns[239:-23])
    taxi_time_od_data = taxi_time_od_data.loc[:,col_name]

    # 归一化
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    taxi_time_od_data = taxi_time_od_data.apply(max_min_scaler, axis=1)

    taxi_time_od_data.to_csv(taxi_ts_path)