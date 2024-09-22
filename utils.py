from tkinter import filedialog, messagebox
import rasterio
from rasterstats import zonal_stats
import datetime
import geopandas as gpd
import json
import os
import pandas as pd


# Define extraction functions

def processing_raster(name, method, clusters, filepath=None):
    if filepath is None:
        messagebox.showinfo('Demand Mapping', 'Select the ' + name + ' map')
        filepath = filedialog.askopenfilename(filetypes=(("rasters", "*.tif"), ("all files", "*.*")))
    raster = rasterio.open(filepath)
    
    clusters = zonal_stats(
        clusters,
        raster.name,
        stats=[method],
        prefix=name, geojson_out=True, all_touched=True)
    
    print(datetime.datetime.now())
    return clusters

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

def spatialjoinvectors(name, column, admin, crs, clusters, val, filepath=None, str=None):
    if filepath is None:
        messagebox.showinfo('Demand Mapping', 'Select the ' + name + ' map')
        filepath = filedialog.askopenfilename(filetypes = (("shapefile","*.shp"),("all files","*.*")))
    # points=gpd.read_file(filedialog.askopenfilename(filetypes = (("shapefile","*.shp"),("all files","*.*"))))
    # points=gpd.read_file(filedialog.askopenfilename(filetypes = (("all files","*.*"),)))
    points=gpd.read_file(filepath)
    points.head(5)
    
    points_clip = gpd.clip(points, admin)
    points_clip.crs = {'init' :'epsg:4326'}
    points_proj=points_clip.to_crs({ 'init': crs})
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
