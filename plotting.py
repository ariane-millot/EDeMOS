# Mapping / Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
from shapely.geometry.point import Point

import geopandas as gpd
import numpy as np


def plot_sector_consumption_map(grid_gdf, col_to_plot, app_config, admin_gdf_param, region_gdf_param, sector_name,
                           lines_gdf=None, fig_size=(25, 15), title = True,
                                plot_as_dots=False, dot_markersize=50):
    print(f"Plotting {sector_name} Consumption map...")

    if col_to_plot not in grid_gdf.columns:
        print(f"Warning: Column '{col_to_plot}' not found for {sector_name} Consumption map.")
        return

    grid_display = grid_gdf.copy()

    col_lower = col_to_plot.lower()
    if 'gwh' in col_lower:
        unit_label = 'GWh'
    elif 'kwh' in col_lower:
        grid_display[col_to_plot] = grid_display[col_to_plot] / 10**3 # kWh to MWh for display
        unit_label = 'MWh'
    else:
        print(f"Warning: Could not detect a known unit (kWh, MWh, GWh) in column name '{col_to_plot}'. Plotting raw values.")

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    # Create a series of only the positive values from the column
    positive_values = grid_display[col_to_plot][grid_display[col_to_plot] > 0]
    data_min = positive_values.min()
    data_max = positive_values.max()

    # --- Extend the normalization range to the nearest powers of 10 ---
    # This ensures the colorbar has a clean, readable range (e.g., 10, 100, 1000)
    log_min_power = np.floor(np.log10(data_min))
    log_max_power = np.ceil(np.log10(data_max))

    # Define the new vmin and vmax for the colormap normalization
    norm_vmin = 10**log_min_power
    norm_vmax = 10**log_max_power

    # Generate ticks for every power of 10 in the new, extended range
    ticks = [10**i for i in range(int(log_min_power), int(log_max_power) + 1)]

    plot_kwargs = {
        "ax": ax,
        "column": col_to_plot,
        "cmap": "Reds",
        "legend": True,
        "alpha": 0.9,
        "norm": colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax),
        "legend_kwds": {
            "label": sector_name + " Consumption (" + unit_label + ")",
            "ticks": ticks,
        }
    }

    # Sort to ensure highest values are plotted on top if there is overlap
    sorted_grid = grid_display.sort_values(col_to_plot, ascending=True)

    if plot_as_dots:
        # Convert geometries to centroids for dot plotting
        # Warning: We filter > 0 here to avoid plotting thousands of empty dots
        dots_gdf = sorted_grid[sorted_grid[col_to_plot] > 0].copy()

        # Use centroids. Note: specific warning suppression might be needed depending on CRS
        dots_gdf['geometry'] = dots_gdf.geometry.centroid

        dots_gdf.plot(markersize=dot_markersize, **plot_kwargs)
    else:
        # Standard grid map
        sorted_grid.plot(**plot_kwargs)

    # grid_display.sort_values(col_to_plot, ascending=True).plot(
    #     ax=ax, column=col_to_plot, cmap="Reds", legend=True, alpha=0.9,
    #     norm=colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax),
    #     legend_kwds={
    #     "label": sector_name + " Consumption (" + unit_label + " per cell)",
    #     "ticks": ticks,  # Use our manually created ticks
    #     }
    # )

    if admin_gdf_param is not None: admin_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='grey', facecolor='None', alpha=0.6)
    if region_gdf_param is not None: region_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='grey', facecolor='None', alpha=0.2)
    if lines_gdf is not None:
        lines_gdf = lines_gdf.to_crs(admin_gdf_param.crs)
        lines_gdf = gpd.clip(lines_gdf, admin_gdf_param)
        lines_gdf.plot(ax=ax, edgecolor='purple', color='purple', alpha=0.4)

    ax.set_aspect('equal', 'box')
    plt.rcParams.update({'font.size': 12})
    ax.tick_params(axis='both', which='major', labelsize=12)
    if app_config.AREA_OF_INTEREST == 'COUNTRY':
        region_title = app_config.COUNTRY
    else:
        region_title = app_config.AREA_OF_INTEREST
    if title:
        ax.set_title(f'{sector_name} Electricity Consumption in {region_title} ({unit_label})')

    # Scalebar
    points = gpd.GeoSeries([Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=app_config.CRS_WGS84)  # Geographic WGS 84 - degrees
    points = points.to_crs(32619)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    # distance_meters = points.iloc[0].distance(points.iloc[1])
    scalebar = ScaleBar(distance_meters, dimension="si-length", location='lower left', length_fraction=0.1, width_fraction=0.001, units='m', color='black')
    ax.add_artist(scalebar)

    suffix = "_dots" if plot_as_dots else ""
    plt.savefig(app_config.OUTPUT_DIR / f'map_{sector_name}_demand_log_{app_config.COUNTRY}{suffix}.png', bbox_inches='tight')
    # plt.show()
