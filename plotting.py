# Mapping / Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.colors import LogNorm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
from shapely.geometry.point import Point

import geopandas as gpd


def plot_sector_consumption_map(grid_gdf, col_to_plot, app_config, admin_gdf_param, region_gdf_param, sector_name,
                           lines_gdf=None, fig_size=(25, 15)):
    print(f"Plotting {sector_name} Consumption map...")

    if col_to_plot not in grid_gdf.columns:
        print(f"Warning: Column '{col_to_plot}' not found for {sector_name} Consumption map.")
        return

    grid_display = grid_gdf.copy()
    grid_display[col_to_plot] = grid_display[col_to_plot] / 10**3 # kWh to MWh for display

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    v_max = grid_display[col_to_plot].max() if not grid_display[col_to_plot].empty else 1e6
    # v_min = grid_display[col_to_plot].min() if not grid_display[col_to_plot].empty else 1e-3
    # if v_min <= 0 : v_min = 1e-9
    # Create a series of only the positive values from the column
    positive_values = grid_display[col_to_plot][grid_display[col_to_plot] > 0]
    # If there are any positive values, find the minimum. Otherwise, use a default small number.
    v_min = positive_values.min() if not positive_values.empty else 1e-3

    grid_display.sort_values(col_to_plot, ascending=True).plot(
        ax=ax, column=col_to_plot, cmap="Reds", legend=True, alpha=0.9,
        norm=colors.LogNorm(vmin=v_min, vmax=v_max),
        legend_kwds={"label": sector_name + " Consumption (MWh per cell)"})

    if admin_gdf_param is not None: admin_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='brown', facecolor='None', alpha=0.6)
    if region_gdf_param is not None: region_gdf_param.to_crs(grid_gdf.crs).plot(ax=ax, edgecolor='brown', facecolor='None', alpha=0.2)
    if lines_gdf is not None:
        lines_gdf = lines_gdf.to_crs(admin_gdf_param.crs)
        lines_gdf = gpd.clip(lines_gdf, admin_gdf_param)
        lines_gdf.plot(ax=ax, edgecolor='purple', color='purple', alpha=0.4)

    ax.set_aspect('equal', 'box')
    ax.set_title(f'{sector_name} Electricity Consumption in {app_config.AREA_OF_INTEREST} ({col_to_plot}, MWh)')

    # Scalebar
    points = gpd.GeoSeries([Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=app_config.CRS_WGS84)  # Geographic WGS 84 - degrees
    points = points.to_crs(32619)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    # distance_meters = points.iloc[0].distance(points.iloc[1])
    scalebar = ScaleBar(distance_meters, dimension="si-length", location='lower left', length_fraction=0.1, width_fraction=0.001, units='m', color='black')
    ax.add_artist(scalebar)

    plt.savefig(app_config.OUTPUT_DIR / f'map_{sector_name}_demand_log_{app_config.COUNTRY}.png', bbox_inches='tight')
    # plt.show()
