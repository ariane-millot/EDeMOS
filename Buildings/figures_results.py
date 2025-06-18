# Mapping / Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
from shapely.geometry.point import Point

import os
import geopandas as gpd


def plot_residential_demand_map(grid_gdf, app_config, admin_gdf_param, region_gdf_param):
    print("Plotting Residential Demand map...")
    col_to_plot = app_config.COL_RES_ELEC_KWH_METH2_SCALED

    if col_to_plot not in grid_gdf.columns:
        print(f"Warning: Column '{col_to_plot}' not found for Residential Demand map.")
        return

    grid_display = grid_gdf.copy()
    grid_display[col_to_plot] = grid_display[col_to_plot] / 10**6 # kWh to GWh for display

    fig, ax = plt.subplots(figsize=(25, 15))
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    v_min = grid_display[col_to_plot].min() if not grid_display[col_to_plot].empty else 1e-9 # Adjusted for GWh
    v_max = grid_display[col_to_plot].max() if not grid_display[col_to_plot].empty else 1e-3 # Adjusted for GWh
    if v_min <= 0 : v_min = 1e-9

    grid_display.sort_values(col_to_plot, ascending=True).plot(
        ax=ax, column=col_to_plot, cmap="Reds", legend=True, alpha=0.9,
        norm=colors.LogNorm(vmin=v_min, vmax=v_max),
        legend_kwds={"label": "Residential Consumption (GWh per cell)"})

    if admin_gdf_param is not None: admin_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='brown', facecolor='None', alpha=0.6)
    if region_gdf_param is not None: region_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='brown', facecolor='None', alpha=0.2)

    ax.set_aspect('equal', 'box')
    ax.set_title(f'Residential Electricity Consumption in {app_config.AREA_OF_INTEREST} ({col_to_plot}, GWh)')

    points = gpd.GeoSeries([Point(-73.5, 40.5), Point(-74.5, 40.5)], crs="EPSG:4326").to_crs(grid_gdf.crs)
    distance_meters = points.iloc[0].distance(points.iloc[1])
    scalebar = ScaleBar(distance_meters, dimension="si-length", location='lower left', length_fraction=0.1, width_fraction=0.001, units='m', color='black')
    ax.add_artist(scalebar)

    plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_residential_demand_log{app_config.COUNTRY}.png', bbox_inches='tight')
    # plt.show()


def plot_service_demand_map(grid_gdf, app_config):
    print("Plotting Service Demand map...")
    col_to_plot = app_config.COL_SER_ELEC_KWH_FINAL
    if col_to_plot in grid_gdf.columns:
        grid_display = grid_gdf.copy()
        grid_display[col_to_plot] = grid_display[col_to_plot] / 1000 # kWh to MWh

        fig, ax = plt.subplots(figsize=(25, 15))
        v_min = grid_display[col_to_plot].min() if not grid_display[col_to_plot].empty else 1e-6 # Adjusted for MWh
        v_max = grid_display[col_to_plot].max() if not grid_display[col_to_plot].empty else 1.0   # Adjusted for MWh
        if v_min <= 0 : v_min = 1e-6

        grid_display.sort_values(col_to_plot, ascending=True).plot(
            ax=ax, column=col_to_plot, cmap="Reds", legend=True, alpha=0.9,
            legend_kwds={"label": "Service Consumption (MWh per cell)"})
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Service Electricity Consumption in {app_config.AREA_OF_INTEREST} (MWh)')
        plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_service_demand_log{app_config.COUNTRY}.png', bbox_inches='tight')
        # plt.show()
    else:
        print(f"Warning: Column '{col_to_plot}' not found for Service Demand map.")