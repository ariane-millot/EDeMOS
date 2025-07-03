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

    # Check if all regions have at least one cell in grid
    # This approach assumes the grid_gdf already has a column with region names.
    print("Validating that all regions have cells in the pre-processed grid...")
    region_col = app_config.ADMIN_REGION_COLUMN_NAME

    if region_col not in grid_gdf.columns:
        raise KeyError(f"Error: The required region column '{region_col}' was not found in the hexagon grid file.")

    # Get the set of unique regions present in the grid file
    regions_in_grid = set(grid_gdf[region_col].unique())

    # Find which of our target regions are not present in the grid
    missing_regions = set(regions_list) - regions_in_grid

    if missing_regions:
        # If there are any missing regions, raise an error with a descriptive message.
        error_message = (
            f"Error: The following regions were not found in the grid's attribute table: "
            f"{sorted(list(missing_regions))}. "
            "Please check the grid file."
        )
        raise ValueError(error_message)
    else:
        print("Validation successful: All target regions are present in the grid file.")

    return regions_list, admin_gdf, region_gdf, grid_gdf