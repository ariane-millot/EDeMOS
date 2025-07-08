import rasterio
from rasterstats import zonal_stats
import datetime
import geopandas as gpd
import json
import os
import pandas as pd
import config
from exactextract import exact_extract

# Define extraction functions
def processing_raster(name, method, clusters, filepath):
    """
    A high-performance version using the exactextract library.
    'clusters' should be a GeoDataFrame.
    """
    if not filepath:
        raise ValueError("Filepath cannot be None or empty.")
    print("Starting extraction:", datetime.datetime.now())
    # It returns a list of dictionaries, one for each feature.
    results = exact_extract(
        filepath,         # The raster
        clusters,     # The GeoDataFrame
        [method],         # The list of stats to compute
        output='pandas'   # Get the output as a pandas DataFrame
    )

    # The resulting dataframe has columns like 'mean', 'sum', etc.
    # We rename the column to match our desired prefix.
    results.rename(columns={method: f"{name}{method}"}, inplace=True)

    # Join the results back to the original GeoDataFrame
    clusters_with_stats = clusters.join(results)
    print("Extraction done:", datetime.datetime.now())
    return clusters_with_stats


# def processing_raster(name, method, clusters, filepath=None):
#     if not filepath:
#         raise ValueError("Filepath for processing_raster cannot be None or empty.")
#     print("Extraction starting:", datetime.datetime.now())
    # with rasterio.open(filepath) as raster:
    #     clusters = zonal_stats(
    #         clusters,
    #         raster.name,
    #         stats=[method],
    #         prefix=name,
    #         geojson_out=True,
    #         all_touched=False #False #True originally
    #     )
    #
    # print(datetime.datetime.now())
    # return clusters


def finalizing_rasters(workspace, clusters, crs):
    output = workspace + r'\placeholder.geojson'
    with open(output, "w") as dst:
        collection = {
            "type": "FeatureCollection",
            "features": list(clusters)}
        dst.write(json.dumps(collection))
  
    clusters = gpd.read_file(output)
    os.remove(output)
    
    print(datetime.datetime.now())
    return clusters


def convert_features_to_geodataframe(features, crs):
    """
    Directly converts an iterable of GeoJSON-like features into a GeoDataFrame.

    Args:
        features: An iterable (list, generator) of GeoJSON Feature dictionaries.
        crs: The Coordinate Reference System to assign to the new GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame created from the features.
    """
    print("Converting features to GeoDataFrame...")

    gdf = gpd.GeoDataFrame.from_features(features, crs=crs)

    print(datetime.datetime.now())
    return gdf


def spatialjoinvectors(column, admin, crs, clusters, val, filepath=None, str=None):
    if not filepath:
        raise ValueError("Filepath for spatialjoinvectors cannot be None or empty.")    
    # points=gpd.read_file(filedialog.askopenfilename(filetypes = (("shapefile","*.shp"),("all files","*.*"))))
    # points=gpd.read_file(filedialog.askopenfilename(filetypes = (("all files","*.*"),)))
    points=gpd.read_file(filepath)
    points.head(5)
    
    points_clip = gpd.clip(points, admin)

    try:
        if points_clip.crs != config.CRS_WGS84:
            raise KeyError(
                f"Coordinate Reference System (CRS) mismatch: "
                f"Expected '{config.CRS}', but got '{points.crs}'."
            )
        print("CRS matched successfully for points_correct.")
    except KeyError as e:
        print(f"Error: {e}")

    # points_clip.crs = config.CRS_WGS84
    points_proj=points_clip.to_crs(crs)
    if str:
        points_proj[column] = points_proj[column].str.replace(',', '')
        points_proj[column] = points_proj[column].astype(float)     ## added so that the sample mining productions works; you may need to update this as per layer used
    
    gdf_points = points_proj[[column, "geometry"]]
    pointsInPolygon = gpd.sjoin(gdf_points, clusters, how="inner", predicate='within')
    ## Defining operation on the selected data
    if val=="sum":
        group_by_name = pointsInPolygon[["id", column]].groupby(["id"]).sum().reset_index()
    elif val=="mean":
        group_by_name = pointsInPolygon[["id", column]].groupby(["id"]).mean().reset_index()
    clusters = pd.merge(clusters, group_by_name[['id', column]], on='id', how = 'left')
    
    return clusters, points
