import pandas as pd
import geopandas as gpd
import numpy as np
import os

def connect(valid_sv_path_path, taxi_ts_path):
    sv = pd.read_csv(valid_sv_path_path, dtype = {"GID": str}, index_col=0)
    taxi = pd.read_csv(taxi_ts_path, dtype = {"GID": str}, index_col=0)