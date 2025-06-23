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


# General parameters for raster extraction
DEFAULT_RASTER_METHOD_BUILDINGS = "sum"
DEFAULT_RASTER_METHOD_LOCATIONWP = "median"
DEFAULT_RASTER_METHOD_HREA = "mean"
DEFAULT_RASTER_METHOD_RWI = "mean"
DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MAJ = "majority"
DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MEAN = "mean"
DEFAULT_RASTER_METHOD_GDP = "mean"


def extract_raster_data(grid_gdf, app_config, processing_raster_func, convert_features_to_geodataframe):
    """
    Extracts raster data (WorldPop, HREA, RWI, Falchetta Tiers) to the grid cells.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        processing_raster_func: The `utils.processing_raster` function.

    Returns:
        GeoDataFrame: The grid GeoDataFrame with added raster data columns.
    """
    print("Extracting raster data...")
    initial_crs = grid_gdf.crs
    print(grid_gdf.crs)
    # WorldPop Buildings Count
    path_wp_bui_count = os.path.join(app_config.WORLDPOP_PATH, app_config.WP_BUILDINGS_COUNT_TIF)
    grid_gdf = processing_raster_func(
        name="buildings", # prefix for new column, e.g. 'buildingssum'
        method=DEFAULT_RASTER_METHOD_BUILDINGS,
        clusters=grid_gdf,
        filepath=path_wp_bui_count
    )
    print(f"Processed WorldPop Buildings Count.")

    # WorldPop Urban
    path_wp_urban = os.path.join(app_config.WORLDPOP_PATH, app_config.WP_BUILDINGS_URBAN_TIF)
    grid_gdf = processing_raster_func(
        name="locationWP",
        method=DEFAULT_RASTER_METHOD_LOCATIONWP,
        clusters=grid_gdf,
        filepath=path_wp_urban
    )
    print(f"Processed WorldPop Urban.")

    # HREA Lighting
    path_hrea_lighting = os.path.join(app_config.LIGHTING_PATH, app_config.HREA_LIGHTING_TIF)
    grid_gdf = processing_raster_func(
        name="HREA",
        method=DEFAULT_RASTER_METHOD_HREA,
        clusters=grid_gdf,
        filepath=path_hrea_lighting
    )
    print(f"Processed HREA Lighting.")

    # RWI
    path_rwi = os.path.join(app_config.RWI_PATH, app_config.RWI_MAP_TIF)
    grid_gdf = processing_raster_func(
        name="rwi",
        method=DEFAULT_RASTER_METHOD_RWI,
        clusters=grid_gdf,
        filepath=path_rwi
    )
    print(f"Processed RWI.")

    # Falchetta Tiers - Majority
    path_falchetta_tiers = os.path.join(app_config.FALCHETTA_PATH, app_config.FALCHETTA_TIERS_TIF)
    grid_gdf = processing_raster_func(
        name="tiers_falchetta_maj",
        method=DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MAJ,
        clusters=grid_gdf,
        filepath=path_falchetta_tiers
    )
    print(f"Processed Falchetta Tiers (Majority).")

    # Falchetta Tiers - Mean
    grid_gdf = processing_raster_func(
        name="tiers_falchetta_mean",
        method=DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MEAN,
        clusters=grid_gdf,
        filepath=path_falchetta_tiers
    )
    print(f"Processed Falchetta Tiers (Mean).")

    # GDP
    # path_gdp = os.path.join(app_config.GDP_PATH, app_config.GDP_PPP_TIF)
    #         grid_gdf = processing_raster_func(
    #             name="GDP_PPP",
    #             method=app_config.DEFAULT_RASTER_METHOD_GDP, # Assuming this exists in config
    #             clusters=grid_gdf,
    #             filepath=path_gdp
    #         )
    #         print(f"Processed GDP. Columns: {grid_gdf.columns}")

    grid_gdf = convert_features_to_geodataframe(grid_gdf, initial_crs)
    print(grid_gdf.crs)
    # Column renaming based on how processing_raster forms column names (prefix + method)
    rename_map = {
        f"buildings{DEFAULT_RASTER_METHOD_BUILDINGS}": app_config.COL_BUILDINGS_SUM, # e.g. buildingssum
        f"locationWP{DEFAULT_RASTER_METHOD_LOCATIONWP}": app_config.COL_LOCATION_WP, # e.g. locationWPmedian
        f"HREA{DEFAULT_RASTER_METHOD_HREA}": app_config.COL_HREA_MEAN, # e.g. HREAmean
        f"rwi{DEFAULT_RASTER_METHOD_RWI}": app_config.COL_RWI_MEAN, # e.g. rwimean
        f"tiers_falchetta_maj{DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MAJ}": app_config.COL_TIERS_FALCHETTA_MAJ,
        f"tiers_falchetta_mean{DEFAULT_RASTER_METHOD_TIERS_FALCHETTA_MEAN}": app_config.COL_TIERS_FALCHETTA_MEAN,
        # f"GDP_PPP{app_config.DEFAULT_RASTER_METHOD_GDP}" = app_config.COL_GDP_PPP_MEAN,
        f"region_name{app_config.ADMIN_REGION_COLUMN_NAME}": app_config.COL_ADMIN_NAME,
    }

    # Filter out renames for columns not actually present
    actual_rename_map = {k: v for k, v in rename_map.items() if k in grid_gdf.columns}
    grid_gdf.rename(columns=actual_rename_map, inplace=True)
    print(f"Columns after renaming: {grid_gdf.columns}")
    print(grid_gdf.crs)
    # Specific adjustments from original script
    if app_config.COL_TIERS_FALCHETTA_MEAN in grid_gdf.columns:
        grid_gdf[app_config.COL_TIERS_FALCHETTA_MEAN] = grid_gdf[app_config.COL_TIERS_FALCHETTA_MEAN].round().astype(int)
    print(grid_gdf.crs)
    # Fill NaN values: Add 0 values in HREA column when there is none
    if app_config.COL_HREA_MEAN in grid_gdf.columns:
        grid_gdf[app_config.COL_HREA_MEAN] = grid_gdf[app_config.COL_HREA_MEAN].fillna(0)
    else:
        print(f"Warning: Column {app_config.COL_HREA_MEAN} not found for fillna.")

    # Add values in RWI column when there is none
    if app_config.COL_RWI_MEAN in grid_gdf.columns:
        grid_gdf[app_config.COL_RWI_MEAN].fillna(grid_gdf[app_config.COL_RWI_MEAN].mean(numeric_only=True).round(1), inplace=True)
        # print(f"RWI min after fillna: {grid_gdf[app_config.COL_RWI_MEAN].min()}")
        # print(f"RWI max after fillna: {grid_gdf[app_config.COL_RWI_MEAN].max()}")
    else:
        print(f"Warning: Column {app_config.COL_RWI_MEAN} not found for fillna.")
    print(grid_gdf.crs)
    print("Finished extracting and processing raster data.")
    return grid_gdf


def load_un_stats(app_config):
    """
    Loads UN energy balance statistics for residential and services sectors.

    Args:
        app_config: The configuration module.

    Returns:
        tuple: (total_residential_elec_GWh, total_services_elec_GWh)
    """
    print("Loading UN energy balance statistics...")
    eb_path = os.path.join(app_config.ENERGY_BALANCE_PATH, app_config.UN_ENERGY_BALANCE_CSV)
    eb = pd.read_csv(eb_path)

    # Residential electricity
    res_elec_tj_series = eb.loc[
        (eb['COMMODITY'] == app_config.UN_ELEC_CODE) &
        (eb['TRANSACTION'] == app_config.UN_HH_TRANSACTION_CODE) &
        (eb['TIME_PERIOD'] == app_config.UN_ENERGY_YEAR),
        'OBS_VALUE'
    ]
    if res_elec_tj_series.empty:
        raise ValueError(f"No UN residential energy data found for year {app_config.UN_ENERGY_YEAR} with specified codes.")
    res_elec_tj_series = pd.to_numeric(res_elec_tj_series.str.replace(',', '').iloc[0])
    total_residential_elec_GWh = res_elec_tj_series / 3.6
    print(f"Total Residential Energy (UN Stats): {total_residential_elec_GWh:.0f} GWh")

    # Services electricity
    ser_elec_tj_series = eb.loc[
        (eb['COMMODITY'] == app_config.UN_ELEC_CODE) &
        (eb['TRANSACTION'] == app_config.UN_SERVICES_TRANSACTION_CODE) &
        (eb['TIME_PERIOD'] == app_config.UN_ENERGY_YEAR),
        'OBS_VALUE'
    ]
    if ser_elec_tj_series.empty:
        raise ValueError(f"No UN services energy data found for transaction {app_config.UN_SERVICES_TRANSACTION_CODE}.")
    ser_elec_tj_series = pd.to_numeric(ser_elec_tj_series.str.replace(',', '').iloc[0])
    total_services_elec_GWh = ser_elec_tj_series / 3.6
    print(f"Total Services Energy (UN Stats): {total_services_elec_GWh:.0f} GWh")

    return total_residential_elec_GWh, total_services_elec_GWh


def load_census_data(app_config):
    """
    Loads provincial and national level census data from CSV files.

    Args:
        app_config: The configuration module.

    Returns:
        tuple: (df_censusdata, df_nationaldata)
               - df_censusdata: DataFrame with provincial census data.
               - df_nationaldata: DataFrame with national census data.
    """
    print("Loading census data...")
    if app_config.PROVINCE_DATA_AVAILABLE:
        # Provincial census data
        census_path = os.path.join(app_config.RESIDENTIAL_DATA_PATH, app_config.CENSUS_PROVINCE_CSV)
        df_censusdata = pd.read_csv(census_path)
        # Process provincial census data
        data_HH = df_censusdata[['Region', 'Urban', 'Rural','size_HH_urban', 'size_HH_rural']]
        data_HH.rename(columns={'Region': 'region', 'Urban': 'HH_urban', 'Rural': 'HH_rural'}, inplace=True)
        data_HH.set_index('region', inplace=True)
        data_HH['HH_total'] = data_HH['HH_urban'] + data_HH['HH_rural']
        data_HH = data_HH.astype(float)
        print(f"Provincial census data loaded: {df_censusdata.shape}")
    else:
        # National census data
        national_census_path = os.path.join(app_config.RESIDENTIAL_DATA_PATH, app_config.CENSUS_NATIONAL_CSV)
        df_censusdata = pd.read_csv(national_census_path)
        data_HH = df_censusdata[['Urban', 'Rural','size_HH_urban', 'size_HH_rural']]
        print(f"National census data loaded: {df_censusdata.shape}")
    df_censusdata.rename(columns={'Region': 'region'}, inplace=True)
    df_censusdata.set_index('region', inplace=True)
    # return df_censusdata, data_HH
    return data_HH, df_censusdata