"""BaseGridHex

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

"""### Define area of interest"""

area = config.AREA_OF_INTEREST

print(area)

"""### Functions for creating heaxgons"""

from create_hex import*

"""### Import layers to be used"""

## admininstrative boundary
if area == "COUNTRY":
    admin_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_COUNTRY)
    region_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_REGION)
else:
    region_gdf = gpd.read_file(config.ADMIN_PATH / config.ADMIN_GPKG, layer=config.ADMIN_LAYER_REGION)
    region_gdf = region_gdf[region_gdf[region_col_name]==area]
    admin_gdf = region_gdf

print(admin_gdf.crs)

"""### H3 - Hexagon - grid"""

size = config.HEX_SIZE ## resolution info here https://h3geo.org/docs/core-library/restable
hexagons = feat(admin_gdf, size)
hexagons.to_file(config.OUTPUT_DIR / "hex.geojson")

# # Plot basemap
# fig, ax = plt.subplots(figsize=(25, 15))
# hexagons.plot(ax=ax, edgecolor='brown', alpha=0.2)
# admin_gdf.plot(ax=ax, edgecolor='brown', alpha=0.2)
# ax.set_aspect('equal', 'box')

# # Save plot as figure
# #plt.savefig('admin level basemap.png', bbox_inches='tight')

# Clipping to the borders of the admin area
join_left_df = gpd.sjoin(hexagons, region_gdf[[config.ADMIN_REGION_COLUMN_NAME, "geometry"]], how="left")
hexagons = join_left_df[join_left_df[config.ADMIN_REGION_COLUMN_NAME].notnull()]
hexagons = hexagons.drop(columns=['index_right'])

# Plot basemap
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(25, 15))
# fig, ax = plt.subplots(figsize=(4, 3))


# hex_reproj = hexagons.to_crs(32619)  # Convert the dataset to a coordinate
hexagons.plot(ax=ax, edgecolor='brown', alpha=0.2)
admin_gdf.plot(ax=ax, edgecolor='brown', alpha=0.2)
ax.set_aspect('equal', 'box')
# Add latitude and longitude labels
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')

# Compute the distance-per-pixel of the map
# see https://geopandas.org/en/latest/gallery/matplotlib_scalebar.html#Geographic-coordinate-system-(degrees)
assert admin_gdf.crs == 'EPSG:4326'
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

plt.show()
# Save plot as figure
plt.savefig(config.OUTPUT_DIR / 'admin_level_basemap.png', bbox_inches='tight')

"""#### Select base map grid"""

hexagons['id'] = range(1, len(hexagons)+1)

hexagons.head(3)

hexagons.columns

# Export dataframe to csv or gpkg
#hexagons.to_csv(out_path + "\\" + f'h3_grid_at_hex_{size}.csv', index=False)
hexagons.to_file(config.OUTPUT_DIR / f'h3_grid_at_hex_{size}.shp', index=False)
hexagons.to_file(config.OUTPUT_DIR / config.H3_GRID_HEX_SHP, index=False) # file used in the other scripts
admin_gdf.to_file(config.OUTPUT_DIR / f'area_gdf.gpkg', index=False)
admin_gdf.to_file(config.OUTPUT_DIR  / f'area_gdf.geojson', driver='GeoJSON', index=False)

