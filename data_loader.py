import os
import pandas as pd
import geopandas as gpd

def load_initial_data(app_config):
    """
    Loads administrative boundaries, region list, and the base hexagonal grid.

    Args:
        app_config: The configuration module.

    Returns:
        tuple: (regions_list, admin_gdf, region_gdf, grid_gdf)
               - regions_list: List of region names to process.
               - admin_gdf: GeoDataFrame of the country boundary.
               - region_gdf: GeoDataFrame of administrative regions.
               - grid_gdf: GeoDataFrame of the hexagonal grid.
    """
    print("Loading initial data...")

    # Load administrative boundaries
    admin_gpkg_path = os.path.join(app_config.ADMIN_PATH, app_config.ADMIN_GPKG)
    admin_gdf = gpd.read_file(admin_gpkg_path, layer=app_config.ADMIN_LAYER_COUNTRY)
    region_gdf = gpd.read_file(admin_gpkg_path, layer=app_config.ADMIN_LAYER_REGION)
    print(f"Admin boundaries loaded. Country GDF: {admin_gdf.shape}, Region GDF: {region_gdf.shape}")

    # List all regions to process
    regions_list = region_gdf[app_config.ADMIN_REGION_COLUMN_NAME].unique().tolist()

    # Filter region_gdf if area_of_interest is not COUNTRY
    area_of_interest = app_config.AREA_OF_INTEREST
    if area_of_interest != "COUNTRY":
        if area_of_interest not in regions_list:
            raise ValueError(f"Error: The region '{area_of_interest}' is not found in the GeoPackage.")
        regions_list = [area_of_interest]
        region_gdf = region_gdf[region_gdf[app_config.ADMIN_REGION_COLUMN_NAME].isin(regions_list)]
        print(f"Filtered region_gdf for {area_of_interest}: {region_gdf.shape}")

    # Load hexagon grid
    hexagons_path = os.path.join(app_config.OUTPUT_DIR, app_config.H3_GRID_HEX_SHP)
    hexagons = gpd.read_file(hexagons_path)
    grid_gdf = hexagons
    print(f"Hexagon grid loaded: {grid_gdf.shape}")

    return regions_list, admin_gdf, region_gdf, grid_gdf