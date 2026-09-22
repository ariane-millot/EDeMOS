"""BaseGridHex.ipynb


# Part 1. Create base grid with H3

### Import necessary modules
"""

# Spatial
import geopandas as gpd
from geopandas.tools import sjoin

# Mapping / Plotting
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import config

"""### Functions for creating heaxgons"""

from create_hex import*

"""### Define area of interest"""

area = config.AREA_OF_INTEREST
print(area)



"""### Import layers to be used"""

## admininstrative boundary
if area == "COUNTRY":
    admin_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_COUNTRY)
    region_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_REGION)
else:
    region_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_REGION)
    region_gdf = region_gdf[region_gdf[config.ADMIN_REGION_COLUMN_NAME]==area]
    admin_gdf = region_gdf

print(admin_gdf.crs)

grid_output_path = config.OUTPUT_DIR / config.H3_GRID_HEX_SHP

if grid_output_path.exists():
    print(f"Hexagon grid {config.H3_GRID_HEX_SHP} already exists. Skipping regeneration.")
else:
    """### H3 - Hexagon - grid"""
    print("Creating a buffer to ensure full hexagon coverage...")
    # Define buffer distance in meters.
    buffer_distance_meters = config.buffer_distance_meters

    # Store original CRS
    original_crs = admin_gdf.crs
    # Reproject to a projected CRS (e.g., a UTM zone) for accurate buffering in meters.
    admin_gdf_proj = admin_gdf.to_crs(config.CRS_PROJ)

    # Create a single unified geometry for buffering.
    unified_geometry = admin_gdf_proj.union_all()
    # Apply the buffer
    buffered_geometry_proj = unified_geometry.buffer(buffer_distance_meters)

    # Create a new GeoDataFrame for the buffered area
    admin_gdf_buffered_proj = gpd.GeoDataFrame(geometry=[buffered_geometry_proj], crs=config.CRS_PROJ)
    # Reproject the buffered GeoDataFrame back to the original CRS (WGS84)
    admin_gdf_buffered = admin_gdf_buffered_proj.to_crs(original_crs)

    print("Buffer created successfully.")

    size = config.HEX_SIZE ## resolution info here https://h3geo.org/docs/core-library/restable
    hexagons_unclipped = feat(admin_gdf_buffered, size)
    print("Clipping hexagons and attaching region attributes...")
    hexagons = gpd.sjoin(hexagons_unclipped,  region_gdf[[config.ADMIN_REGION_COLUMN_NAME, "geometry"]], how="inner", predicate="intersects")
    hexagons = hexagons.drop(columns=['index_right'])
    hexagons = hexagons.drop(columns=['index'])
    hexagons = hexagons.drop_duplicates(subset='h3_index').reset_index(drop=True)
    hexagons['id'] = range(1, len(hexagons)+1)
    print(hexagons.columns)

    def plot_and_save_map(hexagons, admin_gdf, region_gdf):
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(25, 15))
        hexagons.plot(ax=ax, edgecolor='brown', alpha=0.2)
        admin_gdf.plot(ax=ax, edgecolor='brown', alpha=0.2)
        region_gdf.plot(ax=ax, edgecolor='brown', alpha=0.2)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        # Compute the distance-per-pixel of the map
        # see https://geopandas.org/en/latest/gallery/matplotlib_scalebar.html#Geographic-coordinate-system-(degrees)
        assert admin_gdf.crs == config.CRS_WGS84
        from shapely.geometry.point import Point
        points = gpd.GeoSeries([Point(-73.5, 40.5), Point(-74.5, 40.5)], crs=4326)
        points = points.to_crs(32619)
        distance_meters = points[0].distance(points[1])
        scalebar = ScaleBar(distance_meters, dimension="si-length", location='lower left', length_fraction=0.1, width_fraction=0.001, units='m', color='black')
        ax.add_artist(scalebar)
        plt.savefig(config.OUTPUT_DIR / f'admin_level_basemap_{config.COUNTRY}.png', bbox_inches='tight')
        plt.close()
        print(f"Map saved to {config.OUTPUT_DIR}")

    plot_and_save_map(hexagons, admin_gdf, region_gdf)
    hexagons.to_file(grid_output_path, index=False)