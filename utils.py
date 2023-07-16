from tkinter import filedialog, messagebox
import rasterio
from rasterstats import zonal_stats
import datetime
import geopandas as gpd
import json
import os


# Define extraction functions

def processing_raster(name, method, clusters, filepath=None):
    if filepath is None:
        messagebox.showinfo('Demand Mapping', 'Select the ' + name + ' map')
        filepath = filedialog.askopenfilename(filetypes = (("rasters","*.tif"),("all files","*.*")))
    raster=rasterio.open(filepath)
    
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
