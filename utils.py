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


def spatialjoinvectors(column, admin, crs, clusters, val, filepath=None, convert_str_to_float=False):
    """
    Spatially joins point/vector data with polygon clusters (e.g. H3 hexagons),
    aggregates by 'sum' or 'mean', and merges the result into the clusters GeoDataFrame.

    Args:
        column (str): Attribute column to aggregate.
        admin (gpd.GeoDataFrame): Boundary polygon for clipping.
        crs: Coordinate Reference System to reproject the vector points to.
        clusters (gpd.GeoDataFrame): Polygon grid (must contain an 'id' column).
        val (str): Aggregation operation, 'sum' or 'mean'.
        filepath (str or Path): Path to the point/vector dataset.
        convert_str_to_float (bool): If True, strips commas from the column and converts to float.

    Returns:
        tuple: (clusters, points)
    """
    if not filepath:
        raise ValueError("Filepath for spatialjoinvectors cannot be None or empty.")    

    points = gpd.read_file(filepath)
    points_clip = gpd.clip(points, admin)

    if points_clip.crs != config.CRS_WGS84:
        print(f"Warning: Coordinate Reference System (CRS) mismatch: Expected '{config.CRS_WGS84}', but got '{points_clip.crs}'.")
    else:
        print("CRS matched successfully for points_correct.")

    points_proj = points_clip.to_crs(crs)
    if convert_str_to_float:
        points_proj[column] = points_proj[column].astype(str).str.replace(',', '').astype(float)
    
    gdf_points = points_proj[[column, "geometry"]]
    pointsInPolygon = gpd.sjoin(gdf_points, clusters, how="inner", predicate='within')

    ## Defining operation on the selected data
    if val == "sum":
        group_by_name = pointsInPolygon[["id", column]].groupby(["id"]).sum().reset_index()
    elif val == "mean":
        group_by_name = pointsInPolygon[["id", column]].groupby(["id"]).mean().reset_index()
    else:
        raise ValueError(f"Unsupported aggregation operation '{val}'. Expected 'sum' or 'mean'.")

    clusters = pd.merge(clusters, group_by_name[['id', column]], on='id', how='left')
    
    return clusters, points
