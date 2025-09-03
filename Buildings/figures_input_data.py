# Mapping / Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors

import os
import geopandas as gpd

def plot_buildings_map(grid_gdf, app_config, fig_size=(15, 10)):
    print("Plotting buildings map...")
    fig, ax = plt.subplots(figsize=fig_size)
    grid_gdf.sort_values(app_config.COL_BUILDINGS_SUM, ascending=True).plot(
        ax=ax, column=app_config.COL_BUILDINGS_SUM, cmap="Reds", legend=True, alpha=0.9)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Buildings in {app_config.AREA_OF_INTEREST}')
    plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_buildings_{app_config.COUNTRY}.png', bbox_inches='tight')
    plt.show()

def plot_hrea_map(grid_gdf, app_config, fig_size=(15, 10)):
    print("Plotting HREA map...")
    fig, ax = plt.subplots(figsize=fig_size)
    # grid_filtered = grid_gdf[(grid_gdf[app_config.COL_BUILDINGS_SUM] >= 1000) & (grid_gdf[app_config.PROB_ELEC_COL] <= 0.1)]
    grid_filtered = grid_gdf[grid_gdf[app_config.COL_BUILDINGS_SUM] >= 2]
    if not grid_filtered.empty:
        grid_filtered.sort_values(app_config.PROB_ELEC_COL, ascending=True).plot(
            ax=ax, column=app_config.PROB_ELEC_COL, cmap="Reds", legend=True, alpha=0.9)
    else:
        print("Warning: HREA map filtered grid is empty. Plotting unfiltered.")
        grid_gdf.sort_values(app_config.PROB_ELEC_COL, ascending=True).plot(
            ax=ax, column=app_config.PROB_ELEC_COL, cmap="Reds", legend=True, alpha=0.9, missing_kwds={'color': 'lightgrey'})

    ax.set_aspect('equal', 'box')
    ax.set_title(f'HREA in {app_config.AREA_OF_INTEREST}')
    plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_hrea_{app_config.COUNTRY}.png', bbox_inches='tight')
    plt.show()

def plot_urban_rural_map(grid_gdf, app_config, fig_size=(15, 10)):
    print("Plotting Urban/Rural map...")
    fig, ax = plt.subplots(figsize=fig_size)
    if app_config.COL_LOC_ASSESSED in grid_gdf.columns:
        grid_gdf.sort_values(app_config.COL_LOC_ASSESSED, ascending=True).plot(
            ax=ax, column=app_config.COL_LOC_ASSESSED, cmap="Reds", legend=True, alpha=0.5)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Urban and Rural Areas (WorldPop) in {app_config.AREA_OF_INTEREST}')
        plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_urban_rural{app_config.COUNTRY}.png', bbox_inches='tight')
        plt.show()
    else:
        print(f"Warning: Column '{app_config.COL_LOC_ASSESSED}' not found for Urban/Rural map.")

def plot_line_proximity_map(grid_gdf, app_config, admin_gdf_param, fig_size=(15, 10)):
    print("Plotting Line Proximity map...")
    fig, ax = plt.subplots(figsize=fig_size)
    if app_config.COL_IS_NEAR_ANY_LINE in grid_gdf.columns:
        grid_gdf.sort_values(app_config.COL_IS_NEAR_ANY_LINE, ascending=True).plot(
            ax=ax, column=app_config.COL_IS_NEAR_ANY_LINE, cmap="Reds", legend=True, alpha=0.5,categorical=True,
            legend_kwds={
                'labels': ['Not Near Line', 'Near Line'],
                'title': 'Grid Cell Proximity',
                'loc': 'upper left'
            }
        )

        # try:
        #     if os.path.exists(app_config.MV_LINES_SHP):
        #         mv_lines = gpd.read_file(app_config.MV_LINES_SHP).to_crs(grid_gdf.crs)
        #         mv_lines.plot(ax=ax, edgecolor='blue', alpha=0.2, label='MV Lines')
        #     if os.path.exists(app_config.HV_LINES_SHP):
        #         hv_lines = gpd.read_file(app_config.HV_LINES_SHP).to_crs(grid_gdf.crs)
        #         hv_lines.plot(ax=ax, edgecolor='purple', alpha=0.2, label='HV Lines')
        #     if admin_gdf_param is not None:
        #          admin_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='black', facecolor='none', alpha=0.2)
        #     plt.legend()
        # except Exception as e:
        #     print(f"Could not load or plot transmission lines for context map: {e}")

        ax.set_aspect('equal', 'box')
        ax.set_title(f'Proximity to HV/MV Lines in {app_config.AREA_OF_INTEREST}')
        plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'map_line_proximity{app_config.COUNTRY}.png', bbox_inches='tight')
        # plt.show()
    else:
        print(f"Warning: Column '{app_config.COL_IS_NEAR_ANY_LINE}' not found for Line Proximity map.")

def plot_electrified(grid_gdf, app_config, fig_size=(15, 10)):
    print("Plotting electrified cells on map...")
    fig, ax = plt.subplots(figsize=fig_size)
    if app_config.COL_STATUS_ELECTRIFIED in grid_gdf.columns:
        grid_gdf.sort_values(app_config.COL_STATUS_ELECTRIFIED, ascending=True).plot(
            ax=ax, column=app_config.COL_STATUS_ELECTRIFIED, cmap="Reds", legend=True, alpha=0.5,categorical=True,
            legend_kwds={
                'labels': ['Electrified', 'Not electrified'],
                'title': 'Grid Cell Elec Access',
                'loc': 'upper left'
            }
        )

        ax.set_aspect('equal', 'box')
        ax.set_title(f'Elec Access in {app_config.AREA_OF_INTEREST}')
        plt.savefig(app_config.RESIDENTIAL_OUTPUT_DIR / f'electrified_status_{app_config.COUNTRY}.png', bbox_inches='tight')
        # plt.show()
    else:
        print(f"Warning: Column '{app_config.COL_STATUS_ELECTRIFIED}' not found for Line Proximity map.")