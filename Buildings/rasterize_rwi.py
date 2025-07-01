"""

# Rasterize RWI data

**Original code:** [Alexandros Korkovelos](https://github.com/akorkovelos) <br />
**Support:** [Ariane Millot](https://github.com/ariane-millot), [Martin J. Stringer]() <br />
**Funding:** Imperial College <br />

---------------------------
"""

# Importing necessary modules
import pandas as pd
import numpy as np
import math

import geopandas as gpd
import rasterio
from rasterio import features
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import affine
import os
import importlib

import matplotlib.pyplot as plt

# Check if we are running the notebook directly, if so import config
if __name__ == "__main__":
    import sys
    sys.path.insert(1, '../')
    import config

importlib.reload(config)

out_path = config.RWI_PATH

## RWI layer
rwi_path = config.RWI_PATH
rwi_name = config.RWI_FILE_CSV

## Import Relative Wealth Index | convert to geodf | export as gpkg
rwi = pd.read_csv(rwi_path / rwi_name)
rwi_gdf = gpd.GeoDataFrame(rwi, geometry=gpd.points_from_xy(rwi.longitude, rwi.latitude), crs=config.CRS_WGS84)
filename_without_ext = rwi_name.split(".")[0]
rwi_gdf.to_file(os.path.join(rwi_path, f"{filename_without_ext}_{config.COUNTRY}.gpkg"), driver="GPKG")

# rwi_gdf.to_file(os.path.join(rwi_path,"{c}".format(c=rwi_name.split(".")[0])), driver="GPKG")

# Reproject data to the proper coordinate system for the country
rwi_gdf_proj = rwi_gdf.to_crs(config.CRS_PROJ)    # for Zambia

rwi_gdf.head(2)

# Define rasterizaton function
def rasterize_vector(inD, outFile, field, res=0.1, dtype='float32'):
        ''' Create raster describing a field in the shapefile

        INPUT
        inD [ geopandas dataframe created from join_results ]
        outFile [ string ] - path to output raster file
        [ optional ] field [ string ] - column to rasterize from inD
        [ optional ] res [ number ] - resolution of output raster in units of inD crs
        '''

        # create metadata
        bounds = inD.total_bounds
        # calculate height and width from resolution
        width = math.ceil((bounds[2] - bounds[0]) / res)
        height = math.ceil((bounds[3] - bounds[1]) / res)

        cAffine = affine.Affine(res, 0, bounds[0], 0, res * -1, bounds[3])
        nTransform = cAffine #(res, 0, bounds[2], 0, res * -1, bounds[1])
        cMeta = {'count':1,
                 'crs': inD.crs,
                 'dtype':dtype,
                 'affine':cAffine,
                 'driver':'GTiff',
                 'transform':nTransform,
                 'height':height,
                 'width':width,
                 'nodata': 0}
        inD = inD.sort_values(by=[field], ascending=False)
        shapes = ((row.geometry, row[field]) for idx, row in inD.iterrows())
        with rasterio.open(outFile, 'w', **cMeta) as out:
            burned = features.rasterize(shapes=shapes,
                                        fill=0,
                                        all_touched=True,
                                        out_shape=(cMeta['height'], cMeta['width']),
                                        transform=out.transform,
                                        merge_alg=rasterio.enums.MergeAlg.replace)
            burned = burned.astype(cMeta['dtype'])
            out.write_band(1, burned)

# Rasterize & export geodataframe by calling the function

field = "rwi"    # Field (column) based on which the rasterization will be based
resolution = 2400     # in meters
out_raster_name = 'rwi_map_proj.tif'
out_raster_name_crs = config.RWI_MAP_TIF
outFile = os.path.join(out_path, out_raster_name)

rasterize_vector(rwi_gdf_proj, outFile, field=field, res=resolution)

src = rasterio.open(out_path / out_raster_name)

# getting extent from bounds for proper vizualization
src_extent = np.asarray(src.bounds)[[0,2,1,3]]

plt.figure(figsize = (15,15))
plt.imshow(src.read(1), cmap='inferno', extent=src_extent)

plt.show()

# Define project function

def reproj(input_raster, output_raster, new_crs, factor):
    dst_crs = new_crs

    with rasterio.open(input_raster) as src:
        print(src.crs)
        print(dst_crs)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width*factor, src.height*factor, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


# Provide the input raster and give a name to the output (reprojected) raster
input_raster = out_path / out_raster_name
output_raster = out_path / out_raster_name_crs

# Set target CRS
new_crs = config.CRS_WGS84

# Provide a factor if you want zoomed in/out results; suggest keeping it to one unless fully understanding the implications
factor = 1

# Run function
reproj(input_raster, output_raster, new_crs, factor)