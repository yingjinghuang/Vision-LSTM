import geopandas as gpd
import rasterio as rio
import rasterio.mask as mask
from tqdm import tqdm
import os

def split_image_by_shapefile(image_path, shapefile_path, output_dir):
    """
    Split a GeoTiff image into multiple smaller PNG images based on each feature in a shapefile.

    Parameters:
    image_path (str): Path to the image file.
    shapefile_path (str): Path to the shapefile.
    output_dir (str): Path to the directory to save the output images.

    Returns:
    None
    """
    # Open raster
    input_raster = rio.open(image_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Open the shapefile using GeoPandas
    # gdf = pd.read_pickle(shapefile_path)
    # gdf = gpd.GeoDataFrame(gdf, geometry="geometry")
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.to_crs(input_raster.crs)
    # print(gdf)
    # gdf = gdf.set_index("panoid")

    for i in tqdm(range(gdf.shape[0])):
    # if True:

        try:
            # Use shapefile geometry to clip raster
            clip_res = mask.mask(input_raster, [gdf.iloc[i].geometry.__geo_interface__], all_touched=False, crop=True, nodata=0)
            # Save census tiff
            out_meta = input_raster.meta.copy()
            # 更新数据参数，“crs”参数需要结合自己的实际需求，设定相应的坐标参数
            out_meta.update({"driver": "GTiff",
                            "height": clip_res[0].shape[1],
                            "width": clip_res[0].shape[2],
                            "transform": clip_res[1]
                            }
                        )
            # 保存文件
            with rio.open(os.path.join(output_dir, "{}.tif".format(gdf.iloc[i].GID)), "w", **out_meta) as dest:
                dest.write(clip_res[0])
            # print("Blcok group {} insects".format(i))
        except Exception as e:
            print(e)
            # pass
            # print("Blcok group {} not insects".format(i))