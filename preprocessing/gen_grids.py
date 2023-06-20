import geopandas as gpd
import warnings
import numpy as np
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

def cal_ratio(row, grid_size, uv_boundary):
    if row["intersects"] == 1:
        return gpd.clip(gpd.GeoSeries(uv_boundary), row.geometry)[0].area / pow(grid_size, 2)
    else:
        return 0

def gen_grids(grid_size, boundary_path, ground_truth_path, grids_geojson_path, grids_label_path, threshold=0):

    boundary = gpd.read_file(boundary_path)

    boundary = boundary.to_crs(2381)

    xmin, ymin, xmax, ymax = boundary.total_bounds

    x_range = int((xmax - xmin) / grid_size) + 1
    y_range = int((ymax - ymin) / grid_size) + 1

    polygons = []
    for i in range(x_range):
        for j in range(y_range):
            xmin_grid = xmin + i * grid_size
            xmax_grid = xmin + (i + 1) * grid_size
            ymin_grid = ymin + j * grid_size
            ymax_grid = ymin + (j + 1) * grid_size
            polygons.append(Polygon([(xmin_grid, ymin_grid),
                                    (xmax_grid, ymin_grid),
                                    (xmax_grid, ymax_grid),
                                    (xmin_grid, ymax_grid)]))

    grids = gpd.GeoDataFrame({"geometry": polygons}, crs=2381)

    # 使用地块裁剪有效格网
    grids_valid = grids[grids.geometry.intersects(boundary.unary_union)]

    # 重设 GID
    grids_valid["GID"] = ["G"+str(i).zfill(len(str(grids_valid.index[-1]))) for i in range(grids_valid.shape[0])]
    grids_valid.reset_index(drop=True, inplace=True)

    uv = gpd.read_file(ground_truth_path)
    uv = uv.to_crs(grids_valid.crs)

    uv_boundary = uv["geometry"].unary_union

    # 先筛选出有交集的grid
    grids_valid["intersects"] = grids_valid.intersects(uv_boundary).map({True: 1, False: 0})

    # 计算有城中村的格网，城中村的面积占格网面积的比例
    grids_valid["ratio"] = grids_valid.apply(cal_ratio, args=(grid_size, uv_boundary,), axis=1)
    grids_valid["label"] = grids_valid["ratio"].apply(lambda x: 1 if x > threshold else 0)

    # 保存
    grids_valid = grids_valid.to_crs(4326)
    grids_valid.to_file(grids_geojson_path, driver="GeoJSON")
    grids_valid[["GID", "ratio", "label"]].to_csv(grids_label_path)

if __name__=="__main__":
    boundary_path = "./data/raw/sz_qu.json"
    ground_truth_path = "./data/raw/urbanVillage_sz/urbanVillage_sz_landtruth_2018.shp"
    grid_size = 500

    # output path for grids geojson
    grids_geojson_path = "./data/grids_all_500m/grids_all_500m.geojson"
    # output path for grids labels csv
    grids_label_path = "./data/grids_all_500m/grids_label_all_500m.csv"

    gen_grids(grid_size, boundary_path, ground_truth_path, grids_geojson_path, grids_label_path)
