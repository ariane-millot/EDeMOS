# -*- coding: utf-8 -*-
"""building_demand.ipynb

# Building Demand Method 1 Simplified

#### Brief overview:

The energy demand for each cell is assessed according to the following parameters:
𝐵 Number of buildings
𝑆𝑟𝑒𝑠 Share of res buildings
𝑁 Nb of HH per res buildings
𝑎 Electrified status (probability)
𝐸_𝐻𝐻  Energy consumption per HH
𝑟 Adjustment with RWI

For each cell c, we have 𝐷_𝑐=𝐵_𝑐∗𝑆𝑟𝑒𝑠∗𝑁_𝑐  ∗𝑎_𝑐  ∗𝐸_𝐻𝐻  ∗𝑟_𝑐

### Import necessary modules
"""

# Check if we are running the notebook directly, if so move workspace to parent dir
import sys
import os
currentdir = os.path.abspath(os.getcwd())
if os.path.basename(currentdir) != 'DemandMappingZambia':
  sys.path.insert(0, os.path.dirname(currentdir))
  os.chdir('..')
  print(f'Move to {os.getcwd()}')

### Activate geospatial_env first

# Numeric
import numpy as np
import pandas as pd
import math

# System
import shutil
from IPython.display import display, Markdown, HTML, FileLink, FileLinks

# Spatial
import geopandas as gpd
# import json
# import pyproj
# from shapely.geometry import Point, Polygon, MultiPoint
# from shapely.geometry.polygon import Polygon
# from shapely.geometry import shape, mapping
# from shapely.wkt import dumps, loads
# from shapely.ops import nearest_points
# from shapely.ops import unary_union
from pyproj import CRS
# from osgeo import ogr, gdal, osr
# from rasterstats import zonal_stats
# import rasterio
# from geojson import Feature, Point, FeatureCollection
# import rasterio.fill
# import json
# import fiona
# import h3 as h3

# Mapping / Plotting
# from functools import reduce
# import folium
# from folium.features import GeoJsonTooltip
# from folium.plugins import BeautifyIcon
# from folium.plugins import HeatMap
# import branca.colormap as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
# %matplotlib inline

import importlib
import warnings

import tkinter as tk
from tkinter import filedialog, messagebox
import datetime
# import warnings
# import scipy.spatial
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

pd.options.display.float_format = '{:,.2f}'.format

import config
importlib.reload(config)

from utils import processing_raster, finalizing_rasters, convert_features_to_geodataframe

from Residential.data_loader import load_initial_data, extract_raster_data, load_un_stats, load_census_data

area = config.AREA_OF_INTEREST

"""## Import data

### Load initial data grid
"""

# Load initial data (grid and administrative boundaries)
regions, admin_gdf, region_gdf, grid = load_initial_data(config)
print(grid.crs)

"""### Extract raster values to hexagons"""

# Extract raster data
grid = extract_raster_data(grid, config, processing_raster, convert_features_to_geodataframe)
print(grid.crs)

"""### Extract residential and service demand from UN stats"""

total_residential_elec_GWh, total_services_elec_GWh = load_un_stats(config)

"""### Load Census data"""

data_HH, df_censusdata = load_census_data(config)

"""## Residential electricity consumption assessment

### Step 1: assess the number of HH with access
"""

# Plot the buildings map
fig, ax = plt.subplots(figsize=(15, 10))
grid.sort_values('buildingssum', ascending=True).plot(
    ax=ax, column='buildingssum', cmap="Reds", legend=True, alpha=0.9)
ax.set_aspect('equal', 'box')
# txt = ax.set_title('Buildings in {}'.format(area) )

print(grid['buildingssum'].sum())

# Plot the lighting map
# Create the axis first
fig, ax = plt.subplots(figsize=(15, 10))

# Filter the data
grid_filtered = grid[(grid['buildingssum'] >= 1000) & (grid['HREA'] <= 0.1)]
grid_filtered = grid[(grid['buildingssum'] >= 2)]
# Plot data
grid_filtered.sort_values('HREA', ascending=True).plot(
    ax=ax, column='HREA', cmap="Reds", legend=True, alpha=0.9)
# # Plot data
# grid.sort_values('buildingssum', ascending=True).plot(
#     ax=ax, column='buildingssum', cmap="Blues", legend=True, alpha=0.9)

ax.set_aspect('equal', 'box')
# txt = ax.set_title('HREA in {}'.format(area) )

"""#### Determine location (ruban or rural) of each cell"""

def determine_location_status(grid_gdf, app_config):
    """
    Determines urban/rural status for each grid cell based on WorldPop urban extent data.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid with WorldPop data.
        app_config: The configuration module.

    Returns:
        GeoDataFrame: grid_gdf with an added column for location status.
    """
    print("Determining location status (urban/rural)...")
    if app_config.COL_LOCATION_WP not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_LOCATION_WP}' not found in grid. Available: {grid_gdf.columns.tolist()}")

    grid_gdf[app_config.COL_LOC_ASSESSED] = grid_gdf.apply(
        lambda row: "urban" if row[app_config.COL_LOCATION_WP] == 1 else "rural",
        axis=1
    )
    # Other methods
    # grid["locGHSL"] = grid.apply (lambda row: ("urban" if ((row['SMOD'] == 30) or (row['SMOD'] == 21) or (row['SMOD'] == 22) or (row['SMOD' ] == 23))
#                                              else "rural"), axis=1)
    # grid["locAssessed"] = grid.apply(lambda row: ("urban" if ((row['buildingssum'] > 1000)) # number chosen to get 1 for nb of HH per rural building
#                                              else "rural"), axis=1)
    print(f"'{app_config.COL_LOC_ASSESSED}' column created. Counts: {grid_gdf[app_config.COL_LOC_ASSESSED].value_counts().to_dict()}")
    return grid_gdf

grid = determine_location_status(grid, config)

# map of the urban and rural areas WorldPop
fig2, ax2 = plt.subplots(figsize=(10, 5))
grid.sort_values(config.COL_LOC_ASSESSED, ascending=True).plot(
    ax=ax2, column=config.COL_LOC_ASSESSED, cmap="Reds", legend=True, alpha=0.5)
ax2.set_aspect('equal', 'box')
# txt = ax2.set_title('Urban and rural areas WorldPop in {} '.format(area))

"""#### Determine electrifed status of each cell"""

def determine_electrification_status(grid_gdf, app_config, admin_gdf):
    """
    Determines electrification status of grid cells based on proximity to MV/HV lines
    and HREA (High Resolution Electricity Access) likelihood scores.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        admin_gdf: GeoDataFrame of admin boundaries

    Returns:
        GeoDataFrame: grid_gdf with added columns for line proximity and electrification status.
    """
    print("Determining electrification status...")

    # Load MV and HV lines
    mv_lines_gdf = gpd.read_file(app_config.MV_LINES_SHP)
    hv_lines_gdf = gpd.read_file(app_config.HV_LINES_SHP)
    # print(f"MV lines loaded: {mv_lines_gdf.shape}, HV lines loaded: {hv_lines_gdf.shape}")

    print("--- Initial Data Sanity Check ---")
    target_crs = config.TARGET_CRS_METERS
    print(f"Grid CRS: {grid_gdf.crs} | Shape: {grid_gdf.shape}")
    print(f"Admin Boundary CRS: {admin_gdf.crs} | Shape: {admin_gdf.shape}")
    print(f"MV Lines CRS: {mv_lines_gdf.crs} | Shape: {mv_lines_gdf.shape}")
    print(f"HV Lines CRS: {hv_lines_gdf.crs} | Shape: {hv_lines_gdf.shape}")
    print(f"Target CRS for all operations: {target_crs}\n")
    # --- STEP 1: PROJECT ALL DATA TO THE TARGET CRS ---
    # This ensures all subsequent operations (clip, buffer, sjoin) are in the same
    # projected, meter-based coordinate system.

    print("--- Projecting all data to target CRS ---")
    grid_projected = grid_gdf.to_crs(target_crs)
    admin_projected = admin_gdf.to_crs(target_crs)
    # mv_lines_projected = mv_lines_gdf.to_crs(target_crs)
    # hv_lines_projected = hv_lines_gdf.to_crs(target_crs)

    # Initialize the proximity column to False
    grid_gdf[app_config.COL_IS_NEAR_ANY_LINE] = False

    # Define line types and their specific buffer distances
    lines_to_process = [
        {'name': 'HV Lines', 'gdf': hv_lines_gdf, 'buffer_dist': app_config.HV_LINES_BUFFER_DIST},
        {'name': 'MV Lines', 'gdf': mv_lines_gdf, 'buffer_dist': app_config.MV_LINES_BUFFER_DIST}
    ]

    for line_info in lines_to_process:
        current_lines_gdf = line_info['gdf']
        line_name = line_info['name']
        buffer_dist = line_info['buffer_dist']
        print(f"Processing proximity for {line_name} with buffer {buffer_dist}m...")

        if current_lines_gdf.crs != target_crs:
            lines_for_clip_and_buffer = current_lines_gdf.to_crs(target_crs)
        else:
            lines_for_clip_and_buffer = current_lines_gdf.copy()

        # 1. Clip lines
        clipped_lines_gdf = gpd.clip(lines_for_clip_and_buffer, admin_projected)
        if clipped_lines_gdf.empty:
            print(f"No {line_name} found within the admin boundaries after clipping.")
            continue

        # 2. Buffer
        buffered_lines = clipped_lines_gdf.buffer(buffer_dist)
        if buffered_lines.is_empty.all():
            print(f"Buffer for {line_name} is empty. Skipping spatial join.")
            continue

        # 3. Create a GeoDataFrame of the individual buffers
        buffered_areas_gdf = gpd.GeoDataFrame(geometry=buffered_lines, crs=target_crs)

        # 4. Perform the spatial join.
        # Use an 'inner' join to get only the grid cells that intersect.
        intersecting_grid = gpd.sjoin(grid_projected, buffered_areas_gdf, how='inner', predicate='intersects')

        # Get the unique indices of the original grid cells that are near the lines
        indices_to_update = intersecting_grid.index.unique()

        # 5. Update the 'is_near_any_line' column in the ORIGINAL grid.
        grid_gdf.loc[indices_to_update, app_config.COL_IS_NEAR_ANY_LINE] = True

    print(f"Updated 'is_near_any_line' column. Current counts:")
    print(grid_gdf['is_near_any_line'].value_counts())

    if app_config.PROB_ELEC_COL not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.PROB_ELEC_COL}' not found in grid.")
    if app_config.COL_LOC_ASSESSED not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_LOC_ASSESSED}' not found in grid.")

    # electrified or non-electrified status with thresholds depending on the location
    threshold_map = {'urban': app_config.THRESHOLD_ELEC_ACCESS_URBAN, 'rural': app_config.THRESHOLD_ELEC_ACCESS_RURAL}

    grid_gdf[app_config.COL_STATUS_ELECTRIFIED] = grid_gdf.apply(
        lambda row: "elec" if (
            row[app_config.PROB_ELEC_COL] > threshold_map[row[app_config.COL_LOC_ASSESSED]] and
            row[app_config.COL_IS_NEAR_ANY_LINE]
        ) else "nonelec",
        axis=1
    )
    print(f"'{app_config.COL_STATUS_ELECTRIFIED}' column created. Counts: {grid_gdf[app_config.COL_STATUS_ELECTRIFIED].value_counts().to_dict()}")

    return grid_gdf

grid = determine_electrification_status(grid, config, admin_gdf)

# map of the lines
fig3, ax3 = plt.subplots(figsize=(10, 5))
grid.sort_values('is_near_any_line', ascending=True).plot(
    ax=ax3, column='is_near_any_line', cmap="Reds", legend=True, alpha=0.25)
ax3.set_aspect('equal', 'box')
lines_gdf.plot(ax=ax, edgecolor='purple', color='purple', alpha=0.4)
# txt = ax3.set_title('Lines {} '.format(area))

"""#### Assess number of households per cell"""

def calculate_household_numbers(grid_gdf, app_config, data_HH, regions_list):
    """
    Calculates residential building counts and household numbers per grid cell.

    It distinguishes between urban and rural areas and uses either provincial or
    national level census data based on `app_config.PROVINCE_DATA_AVAILABLE`.
    Calculates total population if provincial data (with household sizes) is used.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        data_HH: DataFrame with census data.
        regions_list: List of region names being processed.

    Returns:
        tuple: (grid_gdf, df_HH_buildings)
               - grid_gdf: Updated grid GeoDataFrame with new household/population columns.
               - df_HH_buildings: DataFrame summarizing household data by region (if provincial).
                                  Returns None if national data is used.
    """
    print("Calculating household numbers...")
    df_HH_buildings = None # Initialize for optional return

    if app_config.PROVINCE_DATA_AVAILABLE:
        data_buildings_list = []
        for region_name in regions_list:
            if app_config.COL_ADMIN_NAME not in grid_gdf.columns:
                 raise KeyError(f"Admin name column '{app_config.COL_ADMIN_NAME}' not found in grid for region filtering.")

            totalBuildings = grid_gdf[grid_gdf[app_config.COL_ADMIN_NAME] == region_name][app_config.COL_BUILDINGS_SUM].sum()
            urbanBuildings = grid_gdf[(grid_gdf[app_config.COL_LOC_ASSESSED] == "urban") & (grid_gdf[app_config.COL_ADMIN_NAME] == region_name)][app_config.COL_BUILDINGS_SUM].sum()
            ruralBuildings = grid_gdf[(grid_gdf[app_config.COL_LOC_ASSESSED] == "rural") & (grid_gdf[app_config.COL_ADMIN_NAME] == region_name)][app_config.COL_BUILDINGS_SUM].sum()

            data_region = {
                'region': region_name,
                'totalBuildings': totalBuildings,
                'urbanBuildings': urbanBuildings if totalBuildings > 0 else 0,
                'ruralBuildings': ruralBuildings if totalBuildings > 0 else 0,
                'shareRuralBuild': ruralBuildings / totalBuildings if totalBuildings > 0 else 0,
                'shareUrbanBuild': urbanBuildings / totalBuildings if totalBuildings > 0 else 0,
            }
            data_buildings_list.append(data_region)

        df_buildings = pd.DataFrame(data_buildings_list)
        df_buildings.set_index('region', inplace=True)
        df_HH_buildings = data_HH.merge(df_buildings, left_on='region', right_on='region', how='left')

        # Warning if mismatch happen
        # To avoid division by zero, we calculate the denominator first
        denominator_rural = app_config.NB_OF_HH_PER_RES_BUILDING_URBAN * df_HH_buildings['urbanBuildings']
        # Initialize the column with zeros
        df_HH_buildings['shareUrbanResBui'] = 0.0
        # Identify rows where the denominator is non-zero
        valid_mask_rural = denominator_rural != 0
        # Calculate the share only for valid rows
        df_HH_buildings.loc[valid_mask_rural, 'shareUrbanResBui'] = df_HH_buildings.loc[valid_mask_rural, 'HH_urban'] / denominator_rural[valid_mask_rural]

        # --- Print data inconsistencies ---
        invalid_mask = ~valid_mask_rural & (df_HH_buildings['HH_rural'] > 0)
        if invalid_mask.any():
            lost_hh = df_HH_buildings.loc[invalid_mask, 'HH_rural'].sum()
            regions_affected = df_HH_buildings.loc[invalid_mask].index.tolist()
            print(
                f"Data Inconsistency: {lost_hh:,.0f} urban households could not be allocated "
                f"because no rural buildings were found in region(s): {regions_affected}. "
                f"Their share is set to 0."
            )

        df_HH_buildings['shareUrbanResBui'] = df_HH_buildings['HH_urban'] / (app_config.NB_OF_HH_PER_RES_BUILDING_URBAN * df_HH_buildings['urbanBuildings'])
        df_HH_buildings['shareRuralResBui'] = df_HH_buildings['HH_rural'] / (app_config.NB_OF_HH_PER_RES_BUILDING_RURAL * df_HH_buildings['ruralBuildings'])

        df_HH_buildings.fillna(0, inplace=True)

        df_HH_buildings['resUrbanBui'] = df_HH_buildings['urbanBuildings'] * df_HH_buildings['shareUrbanResBui']
        df_HH_buildings['resRuralBui'] = df_HH_buildings['ruralBuildings'] * df_HH_buildings['shareRuralResBui']
        df_HH_buildings['resTotalBui'] = df_HH_buildings['resUrbanBui'] + df_HH_buildings['resRuralBui']

        # if not app_config.COL_ADMIN_NAME in df_HH_buildings.index:
        #     raise KeyError(f"Required column '{app_config.COL_ADMIN_NAME}' not found in census data.")

        grid_gdf[app_config.COL_RES_URBAN_BUI] = grid_gdf.apply(
            lambda row: row[app_config.COL_BUILDINGS_SUM] * df_HH_buildings.loc[row[app_config.COL_ADMIN_NAME], 'shareUrbanResBui']
                        if row[app_config.COL_LOC_ASSESSED] == 'urban' else 0, axis=1
        )
        grid_gdf[app_config.COL_RES_RURAL_BUI] = grid_gdf.apply(
            lambda row: row[app_config.COL_BUILDINGS_SUM] * df_HH_buildings.loc[row[app_config.COL_ADMIN_NAME], 'shareRuralResBui']
                        if row[app_config.COL_LOC_ASSESSED] == 'rural' else 0, axis=1
        )
    else:
        totalBuildings = grid_gdf[app_config.COL_BUILDINGS_SUM].sum()
        urbanBuildings = grid_gdf[grid_gdf[app_config.COL_LOC_ASSESSED] == "urban"][app_config.COL_BUILDINGS_SUM].sum()
        ruralBuildings = grid_gdf[grid_gdf[app_config.COL_LOC_ASSESSED] == "rural"][app_config.COL_BUILDINGS_SUM].sum()

        nat_HH_urban = data_HH['Urban'].iloc[0]
        nat_HH_rural = data_HH['Rural'].iloc[0]

        shareResBui_urban = nat_HH_urban / (app_config.NB_OF_HH_PER_RES_BUILDING_URBAN * urbanBuildings) if urbanBuildings > 0 else 0
        shareResBui_rural = nat_HH_rural / (app_config.NB_OF_HH_PER_RES_BUILDING_RURAL * ruralBuildings) if ruralBuildings > 0 else 0

        grid_gdf[app_config.COL_RES_URBAN_BUI] = grid_gdf.apply(
            lambda row: row[app_config.COL_BUILDINGS_SUM] * shareResBui_urban if row[app_config.COL_LOC_ASSESSED] == 'urban' else 0, axis=1
        )
        grid_gdf[app_config.COL_RES_RURAL_BUI] = grid_gdf.apply(
            lambda row: row[app_config.COL_BUILDINGS_SUM] * shareResBui_rural if row[app_config.COL_LOC_ASSESSED] == 'rural' else 0, axis=1
        )

    grid_gdf[app_config.COL_RES_BUI] = grid_gdf[app_config.COL_RES_URBAN_BUI] + grid_gdf[app_config.COL_RES_RURAL_BUI]
    grid_gdf[app_config.COL_HH_URBAN] = (grid_gdf[app_config.COL_RES_URBAN_BUI] * app_config.NB_OF_HH_PER_RES_BUILDING_URBAN).fillna(0)
    grid_gdf[app_config.COL_HH_RURAL] = (grid_gdf[app_config.COL_RES_RURAL_BUI] * app_config.NB_OF_HH_PER_RES_BUILDING_RURAL).fillna(0)
    grid_gdf[app_config.COL_HH_TOTAL] = grid_gdf[app_config.COL_HH_URBAN] + grid_gdf[app_config.COL_HH_RURAL]

    if app_config.PROVINCE_DATA_AVAILABLE:
        if not all(f"size_HH_{loc}" in data_HH.columns for loc in ["urban", "rural"]):
            raise KeyError("Missing 'size_HH_urban' or 'size_HH_rural' in census data for population calculation.")

        get_size_HH = lambda row: data_HH.loc[row[app_config.COL_ADMIN_NAME], f"size_HH_{row[app_config.COL_LOC_ASSESSED]}"] \
                                  if row[app_config.COL_ADMIN_NAME] in data_HH.index else np.nan
        grid_gdf[app_config.COL_POP_URBAN] = grid_gdf[app_config.COL_HH_URBAN] * grid_gdf.apply(get_size_HH, axis=1)
        grid_gdf[app_config.COL_POP_RURAL] = grid_gdf[app_config.COL_HH_RURAL] * grid_gdf.apply(get_size_HH, axis=1)
        grid_gdf[app_config.COL_POPULATION] = grid_gdf[app_config.COL_POP_URBAN] + grid_gdf[app_config.COL_POP_RURAL]
        total_population = grid_gdf[app_config.COL_POPULATION].sum()
        print(f"Total population calculated: {total_population:,.0f}")

    print("Finished calculating household numbers.")
    return grid_gdf, df_HH_buildings

grid, df_HH_buildings = calculate_household_numbers(grid, config, data_HH, regions)

df_HH_buildings

"""#### Assess number of households per cell with access to electricity"""

def estimate_hh_with_access(grid_gdf, app_config, df_HH_buildings, data_HH):
    """
    Estimates the number of households with electricity access and calculates access rates.

    Updates grid_gdf with columns for households with and without access.
    If provincial data is available,
    this DataFrame is updated with regional access summaries and saved to a CSV.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        df_HH_buildings: DataFrame for regional household summaries.

    Returns:
        tuple: (grid_gdf, df_HH_buildings)
               - grid_gdf: Updated grid GeoDataFrame.
               - df_HH_buildings: Updated regional summary DataFrame (or None).
    """
    print("Estimating households with access...")

    # Ensure required columns exist
    required_cols = [app_config.COL_HH_URBAN, app_config.COL_HH_RURAL,
                     app_config.PROB_ELEC_COL, app_config.COL_STATUS_ELECTRIFIED]
    for col in required_cols:
        if col not in grid_gdf.columns:
            raise KeyError(f"Required column '{col}' not found in grid_gdf for HH access calculation.")

    grid_gdf[app_config.COL_HH_WITH_ACCESS_URB] = (
        grid_gdf[app_config.COL_HH_URBAN] *
        grid_gdf[app_config.PROB_ELEC_COL] *
        (grid_gdf[app_config.COL_STATUS_ELECTRIFIED] == 'elec') *
        app_config.CORRECTION_FACTOR_URBAN_HH_ACCESS
    )
    grid_gdf[app_config.COL_HH_WITH_ACCESS_RUR] = (
        grid_gdf[app_config.COL_HH_RURAL] *
        grid_gdf[app_config.PROB_ELEC_COL] *
        (grid_gdf[app_config.COL_STATUS_ELECTRIFIED] == 'elec')
    )
    grid_gdf[app_config.COL_HH_WITH_ACCESS] = grid_gdf[app_config.COL_HH_WITH_ACCESS_URB] + grid_gdf[app_config.COL_HH_WITH_ACCESS_RUR]

    # HH without access
    grid_gdf[app_config.COL_HH_WO_ACCESS_URB] = grid_gdf[app_config.COL_HH_URBAN] - grid_gdf[app_config.COL_HH_WITH_ACCESS_URB]
    grid_gdf[app_config.COL_HH_WO_ACCESS_RUR] = grid_gdf[app_config.COL_HH_RURAL] - grid_gdf[app_config.COL_HH_WITH_ACCESS_RUR]
    grid_gdf[app_config.COL_HH_WO_ACCESS] = grid_gdf[app_config.COL_HH_WO_ACCESS_URB] + grid_gdf[app_config.COL_HH_WO_ACCESS_RUR]

    if app_config.PROVINCE_DATA_AVAILABLE and df_HH_buildings is not None:
        print("Aggregating HH access data by region...")
        # Aggregate HH with access by region
        totalHHWithAccessUrb = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[app_config.COL_HH_WITH_ACCESS_URB].sum()
        totalHHWithAccessRur = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[app_config.COL_HH_WITH_ACCESS_RUR].sum()
        totalHHWithAccess = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[app_config.COL_HH_WITH_ACCESS].sum()

        df_HH_access_summary = pd.DataFrame({
            app_config.COL_HH_WITH_ACCESS_URB: totalHHWithAccessUrb,
            app_config.COL_HH_WITH_ACCESS_RUR: totalHHWithAccessRur,
            app_config.COL_HH_WITH_ACCESS: totalHHWithAccess,
        })
        df_HH_access_summary.rename_axis('region', inplace=True)
        # Merge with df_HH_buildings
        df_HH_buildings = df_HH_buildings.merge(df_HH_access_summary, left_index=True, right_index=True)

        # Calculate population with access (requires df_censusdata for HH size)
        # This part might be better placed if df_censusdata is passed directly, or HH size is already in df_HH_buildings_optional
        if app_config.COL_POPULATION in grid_gdf.columns: # Check if population was calculated
            get_size_HH = lambda row: data_HH.loc[row[app_config.COL_ADMIN_NAME], f"size_HH_{row[app_config.COL_LOC_ASSESSED]}"] \
                                  if row[app_config.COL_ADMIN_NAME] in data_HH.index else np.nan
            grid_gdf['population_urban_withAccess'] = grid_gdf[app_config.COL_POPULATION] * grid_gdf.apply(get_size_HH, axis=1).replace([np.inf, -np.inf, np.nan], 0)
            grid_gdf['population_rural_withAccess'] = grid_gdf[app_config.COL_POPULATION] * grid_gdf.apply(get_size_HH, axis=1).replace([np.inf, -np.inf, np.nan], 0)
            grid_gdf['population_withAccess'] = grid_gdf['population_urban_withAccess'] + grid_gdf['population_rural_withAccess']
            total_population_withAccess = grid_gdf['population_withAccess'].sum()
            print(f"Total population with access (estimated): {total_population_withAccess:,.0f}")

        # Calculate access rates in df_HH_buildings
        df_HH_buildings['accessRateHH'] = (df_HH_buildings[app_config.COL_HH_WITH_ACCESS] / df_HH_buildings['HH_total']).replace([np.inf, -np.inf, np.nan], 0)
        df_HH_buildings['accessRateHH_urban'] = (df_HH_buildings[app_config.COL_HH_WITH_ACCESS_URB] / df_HH_buildings['HH_urban']).replace([np.inf, -np.inf, np.nan], 0)
        df_HH_buildings['accessRateHH_rural'] = (df_HH_buildings[app_config.COL_HH_WITH_ACCESS_RUR] / df_HH_buildings['HH_rural']).replace([np.inf, -np.inf, np.nan], 0)

        # Add national summary to df_HH_buildings
        if not df_HH_buildings.empty:
            df_sum = df_HH_buildings[[col for col in df_HH_buildings.columns if col != app_config.COL_ADMIN_NAME]].sum(axis=0, numeric_only=True)
            df_sum[app_config.COL_ADMIN_NAME] = 'National'
            # Recalculate rates for National summary
            df_sum['accessRateHH'] = df_sum[app_config.COL_HH_WITH_ACCESS] / df_sum['HH_total']
            df_sum['accessRateHH_urban'] = df_sum[app_config.COL_HH_WITH_ACCESS_URB] / df_sum['HH_urban']
            df_sum['accessRateHH_rural'] = df_sum[app_config.COL_HH_WITH_ACCESS_RUR] / df_sum['HH_rural']
            df_sum = pd.DataFrame(df_sum).T.set_index(app_config.COL_ADMIN_NAME)
            df_HH_buildings = pd.concat([df_HH_buildings, df_sum])

        output_csv_path = os.path.join(app_config.RESIDENTIAL_OUTPUT_DIR, "dataHH_region.csv")
        df_HH_buildings.to_csv(output_csv_path, index=True)
        print(f"Regional HH summary saved to {output_csv_path}")
        print(df_HH_buildings[['accessRateHH','accessRateHH_urban','accessRateHH_rural']].tail())


    print("Finished estimating households with access.")
    return grid_gdf, df_HH_buildings

grid, df_HH_buildings = estimate_hh_with_access(grid, config, df_HH_buildings, data_HH)

"""### Step 2: assess the electricity consumption per HH

#### Method 1: link the energy consumption to rwi through a logistic function
"""

# Normalise the rwi index
rwi_min = grid['rwi'].min()
rwi_max = grid['rwi'].max()
grid['rwi_norm'] = (grid['rwi'] - rwi_min) / (rwi_max - rwi_min)
grid['rwi_norm'].plot.hist()
plt.show()

# Plot of number of HH vs rwi

# Create equally spaced bins for the 'rwi' values
num_groups = 100
min_rwi = grid['rwi_norm'].min()
max_rwi = grid['rwi_norm'].max()
bin_width = (max_rwi - min_rwi) / num_groups
rwi_bins = [min_rwi + i * bin_width for i in range(num_groups + 1)]
rwi_bins_labels = [(rwi_bins[i] + rwi_bins[i])/2 for i in range(num_groups)]

# Group by the bins and sum the 'HH_total' values
grid['rwi_group'] = pd.cut(grid['rwi_norm'], rwi_bins)
result = grid.groupby('rwi_group')['HH_total'].sum()
result.index = result.index.astype(str)
# # Print the result
# print(result)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(result.index, result.values, color='skyblue', edgecolor='black')
# plt.bar(rwi_bins_labels, result.values, color='skyblue', edgecolor='black')
plt.xlabel('RWI Value Groups')
plt.ylabel('Total HH_total')
plt.title('Sum of HH_total by RWI Value Groups')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot of number of HH with access vs rwi

# Group by the bins and sum the 'HHwithAccess' values
result = grid.groupby('rwi_group')['HHwithAccess'].sum()
result.index = result.index.astype(str)
# # Print the result
# print(result)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(result.index, result.values, color='skyblue', edgecolor='black')
plt.xlabel('RWI Value Groups')
plt.ylabel('Total HH with access')
plt.title('Sum of HH with access by RWI Value Groups')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

def calculate_energy_per_hh_method1(grid_gdf, app_config, total_residential_elec_GWh):
    """
    Calculates energy per household using RWI-based logistic function

    This method normalizes the Relative Wealth Index (RWI), then solves for a
    parameter `k` in a logistic function to ensure the total calculated energy
    matches UN statistics. It adds a column for the calculated energy per household.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_residential_energy_gwh: Total national residential energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for energy per household (Method 1).
    """
    print("Calculating energy per HH (Method 1: RWI-logistic)...")

    if app_config.COL_RWI_MEAN not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_RWI_MEAN}' not found for RWI normalization.")

    rwi_min = grid_gdf[app_config.COL_RWI_MEAN].min()
    rwi_max = grid_gdf[app_config.COL_RWI_MEAN].max()
    if (rwi_max - rwi_min) == 0:
        grid_gdf[app_config.COL_RWI_NORM] = 0.5
        print("Warning: RWI min and max are equal. Normalized RWI set to 0.5.")
    else:
        grid_gdf[app_config.COL_RWI_NORM] = (grid_gdf[app_config.COL_RWI_MEAN] - rwi_min) / (rwi_max - rwi_min)

    alpha = app_config.LOGISTIC_E_THRESHOLD / app_config.LOGISTIC_ALPHA_DERIVATION_THRESHOLD - 1

    if app_config.COL_HH_WITH_ACCESS not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_HH_WITH_ACCESS}' not found for fsolve.")

    def func_solve_k(k_var):
        # Calculates total energy based on k_var and compares to UN total.
        e_hh = app_config.LOGISTIC_E_THRESHOLD / (1 + alpha * np.exp(-k_var * grid_gdf[app_config.COL_RWI_NORM]))
        res_energy_assessed = (e_hh * grid_gdf[app_config.COL_HH_WITH_ACCESS]).sum()
        return res_energy_assessed / 1e6 - total_residential_elec_GWh # kWh to GWh

    try:
        k_solution = fsolve(func_solve_k, app_config.LOGISTIC_K_INITIAL_GUESS)
        # print(f"Solved k for logistic function: {k_solution[0]:.4f}")
        k_to_use = k_solution[0]
    except Exception as e:
        print(f"Error solving for k in RWI-logistic method: {e}. Using initial guess: {app_config.LOGISTIC_K_INITIAL_GUESS}")
        k_to_use = app_config.LOGISTIC_K_INITIAL_GUESS

    grid_gdf[app_config.COL_RES_ELEC_PER_HH_LOG] = app_config.LOGISTIC_E_THRESHOLD / (
        1 + alpha * np.exp(-k_to_use * grid_gdf[app_config.COL_RWI_NORM])
    )
    print("Finished calculating energy per HH (Method 1).")
    return grid_gdf, k_to_use

grid, k_to_use = calculate_energy_per_hh_method1(grid, config, total_residential_elec_GWh)

# create the curve linking energy consumption per HH and rwi
rwi_values = rwi_bins # rwi value groups
k = k_to_use  # Adjust this constant for the desired curve steepness
E_threshold = config.LOGISTIC_E_THRESHOLD
alpha = config.LOGISTIC_E_THRESHOLD / config.LOGISTIC_ALPHA_DERIVATION_THRESHOLD - 1
E_HH_values = config.LOGISTIC_E_THRESHOLD / (1 + alpha * np.exp(-k * np.array(rwi_values)))
# print(E_threshold / (1 + alpha * np.exp(-k  * 0)))
# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(rwi_values, E_HH_values, color='skyblue', edgecolor='black')
plt.xlabel('RWI Value Groups')
plt.ylabel('Electricity consumption per household')
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# plt.title('Energy vs. RWI with logistic relationship')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# check that total energy assessed matches the statistics
grid['ResEnergyPerHH_log'] = E_threshold / (1 + alpha * np.exp(-k * grid['rwi_norm']))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.scatter(grid['rwi_norm'], grid['ResEnergyPerHH_log'], color='skyblue', edgecolor='black')
plt.xlabel('normalised RWI')
plt.ylabel('Electricity consumption per household (kWh)')
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# plt.title('Energy vs. RWI with Logarithmic Relationship')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""#### Method 2: use data coming from the DHS survey"""

def calculate_energy_per_hh_method2(grid_gdf, app_config, estimate_energy_func):
    """
    Calculates energy per household using Method 2 (DHS survey-based).

    This method calls an external script (`estimate_energy_rwi_link_national`)
    which processes DHS data to link RWI and electricity access to electricity consumption.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        estimate_energy_func: The imported `estimate_energy_rwi_link_national` function.

    Returns:
        GeoDataFrame: grid_gdf with added column for electricity per household (Method 2).
    """
    print("Calculating energy per HH (Method 2: DHS-based)...")

    dhs_survey_folder = os.path.join(app_config.RESIDENTIAL_DATA_PATH, "DHSSurvey/")

    grid_gdf = estimate_energy_func(
        grid_gdf,
        dhs_survey_folder,
        app_config.FIGURES_DHS_FOLDER,
        make_figure=app_config.DHS_MAKE_FIGURE,
        recalculate_energies=app_config.DHS_RECALCULATE_ENERGIES,
        simulate_cell_groups=app_config.DHS_SIMULATE_CELL_GROUPS,
        recalculate_energy_perhh=app_config.DHS_RECALCULATE_ENERGY_PERHH
    )

    col_elec_rural = 'elec_demand_kWh_rural'
    col_elec_urban = 'elec_demand_kWh_urban'

    if col_elec_rural not in grid_gdf.columns or col_elec_urban not in grid_gdf.columns:
        raise KeyError(f"Expected columns '{col_elec_rural}' or '{col_elec_urban}' not found after DHS estimation.")

    grid_gdf[app_config.COL_RES_ELEC_PER_HH_DHS] = grid_gdf[col_elec_rural] + grid_gdf[col_elec_urban]

    print("Finished calculating energy per HH (Method 2).")
    return grid_gdf

import Residential.HouseholdEnergyUse.estimate_energy_rwi_link_national_new

# 1. Reload the entire module file
importlib.reload(Residential.HouseholdEnergyUse.estimate_energy_rwi_link_national_new)

# 2. After reloading, re-import the specific function to get the updated version
from Residential.HouseholdEnergyUse.estimate_energy_rwi_link_national_new import estimate_energy_rwi_link_national
grid = calculate_energy_per_hh_method2(grid, config, estimate_energy_rwi_link_national)

"""### Step 3: assess electricity consumption per cell"""

def calculate_total_residential_electricity(grid_gdf, app_config, total_residential_energy_gwh):
    """
    Calculates total residential energy per cell for different methods and scales to UN stats.

    This function takes the per-household energy estimates from Method 1 (RWI-logistic)
    and Method 2 (DHS-based), calculates the total energy per grid cell for each method,
    and then scales these totals so that the national aggregate matches UN statistics.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid with per-HH energy estimates.
        app_config: The configuration module.
        total_residential_energy_gwh: Total national residential energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added columns for raw and scaled total residential energy.
    """
    print("Calculating total residential energy and scaling...")

    # Ensure required input columns exist
    required_cols_meth1 = [app_config.COL_RES_ELEC_PER_HH_LOG, app_config.COL_HH_WITH_ACCESS]
    required_cols_meth2 = [app_config.COL_RES_ELEC_PER_HH_DHS, app_config.COL_HH_WITH_ACCESS]

    for col in required_cols_meth1:
        if col not in grid_gdf.columns: raise KeyError(f"Method 1: Column '{col}' not found.")
    for col in required_cols_meth2:
        if col not in grid_gdf.columns: raise KeyError(f"Method 2: Column '{col}' not found.")

    # For method 1, assign electricity per HH where access > 0, else 0
    grid_gdf["ElecPerHH_kWh_meth1"] = grid_gdf.apply(
        lambda row: row[app_config.COL_RES_ELEC_PER_HH_LOG] if row[app_config.COL_HH_WITH_ACCESS] > 0 else 0, axis=1
    )
    # For method 2, the ElectricityPerHH_DHS is already calculated considering access within its script.
    grid_gdf["ElecPerHH_kWh_meth2"] = grid_gdf[app_config.COL_RES_ELEC_PER_HH_DHS]

    methods_map = {
        'meth1': {'per_hh_col': "ElecPerHH_kWh_meth1", 'output_col_scaled': app_config.COL_RES_ELEC_KWH_METH1_SCALED, 'raw_total_col': app_config.COL_RES_ELEC_KWH_METH1},
        'meth2': {'per_hh_col': "ElecPerHH_kWh_meth2", 'output_col_scaled': app_config.COL_RES_ELEC_KWH_METH2_SCALED, 'raw_total_col': app_config.COL_RES_ELEC_KWH_METH2}
    }

    results_beforescaling_summary = {}
    results_afterscaling_summary = {}

    for method_key, details in methods_map.items():
        # Calculate raw total energy per cell (kWh)
        grid_gdf[details['raw_total_col']] = grid_gdf[app_config.COL_HH_WITH_ACCESS] * grid_gdf[details['per_hh_col']]

        # Aggregate by administrative region (e.g., NAME_1) if COL_ADMIN_NAME is present
        if app_config.COL_ADMIN_NAME in grid_gdf.columns:
            regional_sum_gwh = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[details['raw_total_col']].sum() / 10**6 # kWh to GWh
            results_beforescaling_summary[method_key] = regional_sum_gwh
            total_assessed_gwh = regional_sum_gwh.sum()
        else: # No admin column, sum all cells
            total_assessed_gwh = grid_gdf[details['raw_total_col']].sum() / 10**6 # kWh to GWh
            results_beforescaling_summary[method_key] = pd.Series({"National": total_assessed_gwh})


        if total_assessed_gwh == 0:
            print(f"Warning: Total assessed energy for {method_key} is 0. Scaling factor cannot be computed. Scaled energy will be 0.")
            scaling_factor = 0
        else:
            scaling_factor = total_residential_energy_gwh / total_assessed_gwh

        print(f"Method {method_key}: Total Assessed = {total_assessed_gwh:.2f} GWh, UN Stats = {total_residential_energy_gwh:.2f} GWh, Scaling Factor = {scaling_factor:.4f}")

        grid_gdf[details['output_col_scaled']] = grid_gdf[details['raw_total_col']] * scaling_factor

        if app_config.COL_ADMIN_NAME in grid_gdf.columns:
            results_afterscaling_summary[method_key] = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[details['output_col_scaled']].sum() / 10**6
        else:
            results_afterscaling_summary[method_key] = pd.Series({"National": grid_gdf[details['output_col_scaled']].sum() / 10**6})

    print("\nSummary of energy consumption before scaling (GWh):")
    print(pd.DataFrame(results_beforescaling_summary))
    print("\nSummary of energy consumption after scaling (GWh):")
    print(pd.DataFrame(results_afterscaling_summary))

    print("Finished calculating and scaling total residential energy.")
    return grid_gdf

grid = calculate_total_residential_electricity(grid, config, total_residential_elec_GWh)

"""### Compare access rates to Falchetta dataset"""

def compare_access_to_falchetta(grid_gdf, app_config):
    """
    Compares calculated residential energy consumption tiers with Falchetta dataset tiers.

    This function bins calculated per-household energy into tiers and compares the
    distribution of households across these tiers against pre-loaded Falchetta tier data.
    It also performs a similarity analysis between the DHS-based calculated tiers and
    Falchetta's majority tier.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid with energy consumption data.
        app_config: The configuration module.

    Returns:
        GeoDataFrame: grid_gdf, potentially with added columns for tiering/comparison.
    """
    print("Comparing access tiers to Falchetta dataset...")

    def calculate_tier_share_method(data_grid, method_suffix, hh_with_access_col, hh_wo_access_col, category_total_val):
        # Helper for tier share calculation
        tier_col_name = f'tiers_{method_suffix}'
        if tier_col_name not in data_grid.columns:
            # print(f"Warning: Tier column '{tier_col_name}' not found for method '{method_suffix}'.")
            return pd.Series(dtype=float)
        if category_total_val == 0: return pd.Series(dtype=float)

        tier_share = data_grid.groupby(tier_col_name)[hh_with_access_col].sum()
        if 0 in tier_share.index :
            tier_share.loc[0] += data_grid[hh_wo_access_col].sum()
        else:
            tier_share.loc[0] = data_grid[hh_wo_access_col].sum()
        return tier_share.sort_index() / category_total_val

    bins_tiers = app_config.BINS_TIERS_ENERGY
    tier_labels = range(len(bins_tiers) - 1)

    categories_summary = {
        'national': app_config.COL_HH_TOTAL, 'urban': app_config.COL_HH_URBAN, 'rural': app_config.COL_HH_RURAL
    }

    # Falchetta dataset
    for col_type in [app_config.COL_TIERS_FALCHETTA_MAJ, app_config.COL_TIERS_FALCHETTA_MEAN]:
        if col_type in grid_gdf.columns:
            tiers_summary_df = pd.DataFrame()
            for cat_name, total_hh_col in categories_summary.items():
                 if total_hh_col in grid_gdf.columns and grid_gdf[total_hh_col].sum() > 0:
                    cat_sum = grid_gdf.groupby(col_type)[total_hh_col].sum()
                    tiers_summary_df[cat_name] = cat_sum / cat_sum.sum()
            print(f"\nFalchetta Tiers Summary ({col_type}):")
            print(tiers_summary_df.fillna(0))

    # Our methods
    methods_to_compare = {
        'meth1': app_config.COL_RES_ELEC_PER_HH_LOG,
        'meth2': "ElecPerHH_kWh_meth2"
    }
    categories_for_comparison = [
        ('national', app_config.COL_HH_WITH_ACCESS, app_config.COL_HH_WO_ACCESS, app_config.COL_HH_TOTAL),
        ('urban', app_config.COL_HH_WITH_ACCESS_URB, app_config.COL_HH_WO_ACCESS_URB, app_config.COL_HH_URBAN),
        ('rural', app_config.COL_HH_WITH_ACCESS_RUR, app_config.COL_HH_WO_ACCESS_RUR, app_config.COL_HH_RURAL)
    ]

    for method_key, energy_col_name in methods_to_compare.items():
        if energy_col_name not in grid_gdf.columns:
            print(f"Warning: Energy column '{energy_col_name}' for method '{method_key}' not found.")
            continue

        grid_gdf[f'tiers_{method_key}'] = pd.cut(grid_gdf[energy_col_name], bins=bins_tiers, labels=tier_labels, right=False)
        grid_gdf[f'tiers_{method_key}'] = grid_gdf[f'tiers_{method_key}'].fillna(0).astype(int)

        df_tiers_data = pd.DataFrame()
        for cat_name, hh_access_col, hh_no_access_col, total_hh_col in categories_for_comparison:
            if all(c in grid_gdf.columns for c in [hh_access_col, hh_no_access_col, total_hh_col]):
                cat_total_val = grid_gdf[total_hh_col].sum()
                if cat_total_val > 0:
                    tier_share_series = calculate_tier_share_method(grid_gdf, method_key, hh_access_col, hh_no_access_col, cat_total_val)
                    df_tiers_data[cat_name] = tier_share_series

        print(f"\nTier Shares for Method '{method_key}':")
        print(df_tiers_data.fillna(0))

    if f'tiers_meth2' in grid_gdf.columns and app_config.COL_TIERS_FALCHETTA_MAJ in grid_gdf.columns:
        grid_gdf['tiers_DHS_adjusted'] = grid_gdf['tiers_meth2'].where(grid_gdf['tiers_meth2'] != 5, 4)
        grid_gdf['Similarity_Falchetta_DHS'] = grid_gdf['tiers_DHS_adjusted'] == grid_gdf[app_config.COL_TIERS_FALCHETTA_MAJ]
        grid_gdf['Difference_Falchetta_DHS'] = abs(pd.to_numeric(grid_gdf['tiers_DHS_adjusted']) - pd.to_numeric(grid_gdf[app_config.COL_TIERS_FALCHETTA_MAJ]))

        print("\nSimilarity Analysis (Falchetta vs DHS-Method2):")
        print(f"Number of lines with similar tiers: {grid_gdf['Similarity_Falchetta_DHS'].sum()}")
        print(f"Mean difference in tiers: {grid_gdf['Difference_Falchetta_DHS'].mean():.2f}")
        print(f"Median difference in tiers: {grid_gdf['Difference_Falchetta_DHS'].median():.2f}")

    print("Finished Falchetta comparison.")
    return grid_gdf

grid = compare_access_to_falchetta(grid, config)

"""### Final grid"""

print(grid.columns)
grid.to_csv(config.RESIDENTIAL_OUTPUT_DIR / 'data_res.csv')

if 'rwi_group' in grid.columns:
    grid = grid.drop('rwi_group', axis=1)
if 'tiers_DHS' in grid.columns:
    grid = grid.drop('tiers_DHS', axis=1)
if 'bin_labels' in grid.columns:
    grid = grid.drop('bin_labels', axis=1)
grid.to_file(config.RESIDENTIAL_OUTPUT_DIR / 'res_energy_map.shp', index=False)
grid.head(3)

"""### Map residential results"""

resultRes = 'ResElec_kWh_meth2_scaled'
grid[resultRes] = grid[resultRes]/10**6

# Plot the demand map with a log scale value
# Create the axis first
# sns.set_theme('poster')
# sns.set_style('white')
fig, ax = plt.subplots(figsize=(25, 15))

# Add latitude and longitude labels
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')

# Plot data
grid.sort_values(resultRes, ascending=True).plot(
    ax=ax, column=resultRes, cmap="Reds", legend=True, alpha=0.9, norm=colors.LogNorm(vmin = 1e-6, vmax=grid[resultRes].max()),
    legend_kwds={"label": "Consumption in kWh"}) #, "orientation": "horizontal"})

# admin_gdf.plot(ax=ax, edgecolor='brown', color='None', alpha=0.6)
# region_gdf.plot(ax=ax, edgecolor='brown', color='None', alpha=0.2)
# transmission lines
# lines_gdf.plot(ax=ax, edgecolor='purple', color='purple', alpha=0.4)
# MV_lines_gdf.plot(ax=ax, edgecolor='purple', color='purple', alpha=0.05)

ax.set_aspect('equal', 'box')
# txt = ax.set_title('Electricity consumption in the residential sector in {} (kWh)'.format(area) )
# txt = ax.set_title('Electricity consumption in the residential sector (kWh)' )

# print(grid.crs)

# Compute the distance-per-pixel of the map
# see https://geopandas.org/en/latest/gallery/matplotlib_scalebar.html#Geographic-coordinate-system-(degrees)
# assert grid.crs == 'EPSG:4326'
from shapely.geometry.point import Point
points = gpd.GeoSeries(
    [Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326
)  # Geographic WGS 84 - degrees
points = points.to_crs(32619)  # Projected WGS 84 - meters
distance_meters = points[0].distance(points[1])

# Add a scale bar
scalebar = ScaleBar(
    distance_meters,
    dimension="si-length",
    location='lower left',
    length_fraction=0.1,
    width_fraction=0.001,
    units='m',
    color='black',
    fixed_value=None
)

ax.add_artist(scalebar)

# Save plot as figure
plt.savefig(config.RESIDENTIAL_OUTPUT_DIR / 'map_res_log.png', bbox_inches='tight')

"""# Services

## Electricity consumption based on number of buildings with access
"""

def calculate_service_buildings_elec(grid_gdf, app_config, total_services_elec_gwh):
    """
    Calculates services electricity demand based on the number of accessible service buildings.

    It estimates the number of service buildings, then those with access, and
    distributes the total national services electricity amongst them.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_elec_gwh: Total national services energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for building-based services energy.
    """
    print("Calculating services electricity (building-based)...")

    if not all(col in grid_gdf.columns for col in [app_config.COL_BUILDINGS_SUM, app_config.COL_RES_BUI, app_config.PROB_ELEC_COL]):
        raise KeyError("One or more required columns for service building electricity calculation are missing.")

    grid_gdf[app_config.COL_SER_BUI] = grid_gdf[app_config.COL_BUILDINGS_SUM] - grid_gdf[app_config.COL_RES_BUI]
    grid_gdf[app_config.COL_SER_BUI_ACC] = grid_gdf[app_config.COL_SER_BUI] * grid_gdf[app_config.PROB_ELEC_COL]

    total_ser_bui_with_access = grid_gdf[app_config.COL_SER_BUI_ACC].sum()
    print(f"Total services buildings with estimated access: {total_ser_bui_with_access:,.0f}")

    ser_elec_per_bui_kwh = (total_services_elec_gwh * 1e6) / total_ser_bui_with_access if total_ser_bui_with_access > 0 else 0
    if total_ser_bui_with_access == 0: print("Warning: Total service buildings with access is 0.")

    print(f"Service electricity per accessible building: {ser_elec_per_bui_kwh:,.0f} kWh/building")
    grid_gdf[app_config.COL_SER_ELEC_KWH_BUI] = ser_elec_per_bui_kwh * grid_gdf[app_config.COL_SER_BUI_ACC]

    print("Finished calculating services electricity (building-based).")
    return grid_gdf

grid = calculate_service_buildings_elec(grid, config, total_services_elec_GWh)

"""## Energy consumption based on GDP"""

def calculate_gdp_based_energy(grid_gdf, app_config, total_services_elec_gwh):
    """
    Calculates services energy demand based on GDP data.

    If GDP data is available, it
    distributes total national services energy based on the GDP of each grid cell.
    Otherwise, it sets the GDP-based energy column to zero.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_energy_gwh: Total national services energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for GDP-based services energy.
    """
    print("Calculating services energy (GDP-based)...")

    gdp_col = getattr(app_config, 'COL_GDP_PPP_MEAN', None)
    col_ser_en_gdp = getattr(app_config, 'COL_SER_ELEC_KWH_GDP', 'Ser_elec_kWh_GDP') # Define target column name

    if gdp_col and gdp_col in grid_gdf.columns:
        total_gdp_kdollars = grid_gdf[gdp_col].sum() / 1000
        # print(f"Total GDP: {total_gdp_kdollars:,.0f} k$")
        ser_elec_per_gdp_kwh_per_kdolar = (total_services_elec_gwh * 1e6) / total_gdp_kdollars if total_gdp_kdollars > 0 else 0
        if total_gdp_kdollars == 0: print("Warning: Total GDP is 0.")

        print(f"Service energy per unit of GDP: {ser_elec_per_gdp_kwh_per_kdolar:,.2f} kWh/k$")
        grid_gdf[col_ser_en_gdp] = ser_elec_per_gdp_kwh_per_kdolar * (grid_gdf[gdp_col] / 1000)
        print(f"'{col_ser_en_gdp}' column created/updated.")
    else:
        grid_gdf[col_ser_en_gdp] = 0.0
        print(f"Warning: GDP column '{gdp_col}' not found or not defined. GDP-based service energy set to 0.")

    print("Finished calculating services energy (GDP-based).")
    return grid_gdf

"""## Energy consumption based on employees"""

def calculate_employee_based_energy(grid_gdf, app_config, total_services_elec_gwh, df_censusdata):
    """
    Calculates services energy demand based on the estimated number of employees.

    This function uses DHS survey data for employment rates and working population shares,
    combined with census data for population distribution, to estimate the number of
    employees per grid cell. Total national services energy is then distributed
    among employees with electricity access.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_elec_gwh: Total national services energy (GWh) from UN stats.
        df_censusdata: DataFrame with provincial census data, indexed by region name.

    Returns:
        GeoDataFrame: grid_gdf with added columns for employee counts and employee-based services energy.
    """
    print("Calculating services energy (employee-based)...")

    # Use paths from config for DHS employee data
    path_emp_women = app_config.DHS_EMPLOYEE_WOMEN_CSV
    path_emp_men = app_config.DHS_EMPLOYEE_MEN_CSV
    path_work_pop = app_config.DHS_WORKING_POP_SHARE_CSV

    try:
        data_employee_women = pd.read_csv(path_emp_women, index_col=(0, 1))
        data_employee_men = pd.read_csv(path_emp_men, index_col=(0, 1))
        data_workingpop_share = pd.read_csv(path_work_pop, index_col=(1, 0))
    except FileNotFoundError as e:
        print(f"Error: Employee data file not found: {e}. Skipping employee-based service energy calculation.")
        grid_gdf[app_config.COL_SER_ELEC_KWH_EMP] = 0.0
        return grid_gdf

    # Sum employee shares
    data_employee_women['total_employee_share_women'] = data_employee_women[['professional/technical/managerial', 'clerical', 'sales', 'services', 'skilled manual']].sum(axis=1)
    data_employee_men['total_employee_share_men'] = data_employee_men[['professional/technical/managerial', 'clerical', 'sales', 'services', 'skilled manual']].sum(axis=1)

    # Ensure required columns exist in census data
    required_census_cols = ['Share women', 'size_HH_urban', 'size_HH_rural']
    if not all(col in df_censusdata.columns for col in required_census_cols):
        raise KeyError(f"Missing one or more required columns in census data: {required_census_cols}")

    # --- FIX START: The calculation logic is moved into a more robust helper function ---

    # Make a local copy to avoid SettingWithCopyWarning
    df_censusdata_local = df_censusdata.copy()

    # Define a single, more robust helper function for calculating population
    def calculate_nb_gender(row, gender_type):
        admin_name = row[app_config.COL_ADMIN_NAME]
        loc_status = row[app_config.COL_LOC_ASSESSED]
        hh_total = row[app_config.COL_HH_TOTAL]

        # Get HH size for the specific location type (urban/rural)
        hh_size = df_censusdata_local.loc[admin_name, f"size_HH_{loc_status}"]

        # Determine sex share and working pop share based on gender_type
        if gender_type == 'women':
            regional_sex_share = df_censusdata_local.loc[admin_name, 'Share women']
            working_age_pop_share = data_workingpop_share.loc[('Female', loc_status), '15-49'] / 100
        elif gender_type == 'men':
            regional_sex_share = 1 - df_censusdata_local.loc[admin_name, 'Share women']
            working_age_pop_share = data_workingpop_share.loc[('Male', loc_status), '15-49'] / 100
        else:
            raise ValueError('Unknown gender_type', gender_type)

        return hh_total * hh_size * regional_sex_share * working_age_pop_share

    # Apply the corrected function for both women and men
    print("  Calculating number of men and women (15-49)...")
    grid_gdf['nb_women'] = grid_gdf.apply(calculate_nb_gender, args=('women',), axis=1)
    grid_gdf['nb_men'] = grid_gdf.apply(calculate_nb_gender, args=('men',), axis=1)

    # --- FIX END ---

    # Calculate working women/men (This part was mostly correct)
    def calculate_working_gender(row, sex_col_name, employee_data_df, employee_share_col_name):
        loc_status = row[app_config.COL_LOC_ASSESSED]
        # Normalize the region name to match the index in the employee data
        admin_name_processed = row[app_config.COL_ADMIN_NAME].lower().replace('-', ' ')

        try:
            # Look up the working share from the pre-loaded employee data
            percent_working = employee_data_df.loc[(admin_name_processed, loc_status), employee_share_col_name] / 100
        except KeyError:
            # If a specific region/location combo is missing, default to 0 to avoid errors
            percent_working = 0

        return row[sex_col_name] * percent_working

    print("  Calculating number of working men and women...")
    grid_gdf['nb_women_working'] = grid_gdf.apply(calculate_working_gender, args=('nb_women', data_employee_women, 'total_employee_share_women'), axis=1)
    grid_gdf['nb_men_working'] = grid_gdf.apply(calculate_working_gender, args=('nb_men', data_employee_men, 'total_employee_share_men'), axis=1)

    # This redundant merge is no longer needed as the logic is handled inside the apply functions
    # grid_gdf = grid_gdf.merge(df_censusdata['Share women'], on=app_config.COL_ADMIN_NAME, how='left')

    # Sum up totals
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE] = grid_gdf['nb_women_working'] + grid_gdf['nb_men_working']
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS] = grid_gdf.loc[grid_gdf[app_config.COL_STATUS_ELECTRIFIED] == 'elec', app_config.COL_TOTAL_EMPLOYEE]
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS].fillna(0, inplace=True) # Ensure non-elec rows are 0, not NaN

    total_employee_national_with_access = grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS].sum()
    print(f"Total employees with access: {total_employee_national_with_access:,.0f}")

    if total_employee_national_with_access > 0:
        ser_en_per_employee_kwh = (total_services_elec_gwh * 1e6) / total_employee_national_with_access # kWh / employee
    else:
        print("Warning: Total employees with access is 0. Energy per employee will be 0.")
        ser_en_per_employee_kwh = 0

    print(f"Service electricity per accessible employee: {ser_en_per_employee_kwh:,.2f} kWh/employee")

    # Distribute energy based on employees with access
    grid_gdf[app_config.COL_SER_ELEC_KWH_EMP] = ser_en_per_employee_kwh * grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS]

    print("Finished calculating services energy (employee-based).")
    return grid_gdf

grid = calculate_employee_based_energy(grid, config, total_services_elec_GWh, df_censusdata)

"""## Weighted average of the three assessements"""

# Link between buildings and GDP
plt.scatter(grid['serBUi_Acc'], grid['GDP_PPP'],s=1)

# Add labels to the plot
plt.xlabel('serBUi_Acc')
plt.ylabel('GDP_PPP')

# Set the axis to logarithmic scale
# plt.yscale('log')
# plt.xscale('log')

# Show the plot
plt.show()

threshold_access = 0.1 # lower value than residential because easier to connect services buildings
alpha = 0
beta = 0
gama =1

# compute weighted average
# Create a boolean Series indicating if probElec meets the condition
# condition_met = grid[probElec] >= threshold_access
# Assign the weighted average based on the condition
grid['SElec_kWh_weighted'] = (beta * grid[config.COL_SER_ELEC_KWH_BUI] + gama * grid[config.COL_SER_ELEC_KWH_EMP] )

totalSEn_kWh_weighted = grid['SElec_kWh_weighted'].sum()
grid[config.COL_SER_ELEC_KWH_FINAL] = grid['SElec_kWh_weighted'] / totalSEn_kWh_weighted * total_services_elec_GWh *10**6

"""## Results per region and map"""

services_result = pd.DataFrame()
services_result = grid.groupby('NAME_1')[config.COL_SER_ELEC_KWH_FINAL].sum() / 10**6 # conversion in GWh
pd.options.display.float_format = '{:.2f}'.format
services_result

# Plot the energy consumption in services buildings map
# Create the axis first
fig, ax = plt.subplots(figsize=(25, 15))

# Plot data
grid.sort_values(config.COL_SER_ELEC_KWH_FINAL, ascending=True).plot(
    ax=ax, column=config.COL_SER_ELEC_KWH_FINAL, cmap="Reds", legend=True, alpha=0.9)

ax.set_aspect('equal', 'box')
# txt = ax.set_title('Services electricity consumption in {}'.format(area) )

# Save plot as figure
plt.savefig(out_path + '/services_map' +str(alpha) +str(gama)+'.png', bbox_inches='tight')

grid.to_csv(config.RESIDENTIAL_OUTPUT_DIR / 'dataser.csv')
grid.to_file(config.RESIDENTIAL_OUTPUT_DIR / 'ser_energy_map.shp', index=False)
grid.to_file(config.RESIDENTIAL_OUTPUT_DIR / 'ser_energy_map.geojson', driver='GeoJSON', index=False)
grid.head(3)

total_servicesenergy_scaled = grid.groupby('NAME_1')[config.COL_SER_ELEC_KWH_FINAL].sum()
print ("Services electricity consumption assessed after scaling:")
for region in regions:
    total_servicesenergy_scaled[region] = total_servicesenergy_scaled[region]/10**6  # conversion in GWh
    print (region, f"{total_servicesenergy_scaled[region]:,.1f}", "GWh" )
print (total_servicesenergy_scaled )
print (total_servicesenergy_scaled.sum() )

total_servicesenergy_scaled

"""# Buildings"""

# total_Buienergy_scaled = total_servicesenergy_scaled + total_residentialenergy_scaled
total_Buienergy_scaled = total_servicesenergy_scaled + result_afterscaling['meth2']
print ("Services electricity consumption assessed after scaling:")
for region in regions:
    print (region, f"{total_Buienergy_scaled[region]:,.1f}", "GWh" )
print (total_Buienergy_scaled)

