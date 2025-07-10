import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


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


# General parameters for raster extraction
DEFAULT_RASTER_METHOD_BUILDINGS = "sum"
DEFAULT_RASTER_METHOD_LOCATIONWP = "majority"
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

    # # RWI
    # path_rwi = os.path.join(app_config.RWI_PATH, app_config.RWI_MAP_TIF)
    # grid_gdf = processing_raster_func(
    #     name="rwi",
    #     method=DEFAULT_RASTER_METHOD_RWI,
    #     clusters=grid_gdf,
    #     filepath=path_rwi
    # )
    # print(f"Processed RWI.")

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
        # f"rwi{DEFAULT_RASTER_METHOD_RWI}": app_config.COL_RWI_MEAN, # e.g. rwimean
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
    # Fill NaN values with tier 0
    if app_config.COL_TIERS_FALCHETTA_MEAN in grid_gdf.columns:
        grid_gdf[app_config.COL_TIERS_FALCHETTA_MEAN] = grid_gdf[app_config.COL_TIERS_FALCHETTA_MEAN].fillna(0).round().astype(int)
    print(grid_gdf.crs)
    # Fill NaN values: Add 0 values in HREA column when there is none
    if app_config.COL_HREA_MEAN in grid_gdf.columns:
        grid_gdf[app_config.COL_HREA_MEAN] = grid_gdf[app_config.COL_HREA_MEAN].fillna(0)
    else:
        print(f"Warning: Column {app_config.COL_HREA_MEAN} not found for fillna.")

    # # Add values in RWI column when there is none
    # if app_config.COL_RWI_MEAN in grid_gdf.columns:
    #     grid_gdf[app_config.COL_RWI_MEAN].fillna(grid_gdf[app_config.COL_RWI_MEAN].mean(numeric_only=True).round(1), inplace=True)
    #     # print(f"RWI min after fillna: {grid_gdf[app_config.COL_RWI_MEAN].min()}")
    #     # print(f"RWI max after fillna: {grid_gdf[app_config.COL_RWI_MEAN].max()}")
    # else:
    #     print(f"Warning: Column {app_config.COL_RWI_MEAN} not found for fillna.")
    print(grid_gdf.crs)
    print("Finished extracting and processing raster data.")
    return grid_gdf


def load_rwi_data(grid_gdf, app_config):
    """
    Extracts RWI to the grid cells.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.

    Returns:
        GeoDataFrame: The grid GeoDataFrame with added raster data columns.
    """
    try:
        rwi_df = pd.read_csv(app_config.RWI_PATH / app_config.RWI_FILE_CSV)
    except Exception as e:
        print(f"Error loading RWI CSV file: {e}")
        print("Please ensure the path is correct and the file exists.")
        exit()
    hex_grid = grid_gdf
    print(f"Loaded {len(hex_grid)} hexagons.")
    print(f"Loaded {len(rwi_df)} RWI data points.")

    # --- 3. CREATE A GEODATAFRAME FROM THE RWI CSV ---
    # The RWI data uses 'longitude' and 'latitude' columns.
    # We convert these into Point geometries to create a GeoDataFrame.
    print("Creating GeoDataFrame from RWI data...")
    rwi_gdf = gpd.GeoDataFrame(
        rwi_df,
        geometry=gpd.points_from_xy(rwi_df.longitude, rwi_df.latitude),
        crs=app_config.CRS_WGS84  # The RWI data is in WGS84 (EPSG:4326)
    )

    # --- 4. ENSURE CONSISTENT COORDINATE REFERENCE SYSTEMS (CRS) ---
    if hex_grid.crs != rwi_gdf.crs:
        print(f"Original hexagon CRS: {hex_grid.crs}")
        print(f"RWI points CRS: {rwi_gdf.crs}")
        print("CRS do not match! Error in the workflow!")
        exit()

    # --- 5. PERFORM THE SPATIAL JOIN ---
    # This operation will associate each RWI point with the hexagon it falls within.
    # 'op'='within' finds which points are within which polygons.
    # 'how'='inner' keeps only the points that fall within a hexagon.
    print("Performing spatial join...")
    # joined_gdf = gpd.sjoin(rwi_gdf, hex_grid, how="inner", predicate="within")
    joined_gdf = gpd.sjoin(rwi_gdf, hex_grid, how="inner", predicate="intersects")
    # Store the count BEFORE deduplication
    count_before = len(joined_gdf)
    # If a point intersects two hexagons, it will create two rows. We only want one.
    # We drop duplicates based on the point's original index.
    joined_gdf = joined_gdf.drop_duplicates(subset=['latitude', 'longitude'])
    # The result 'joined_gdf' contains data for each point, plus the index ('index_right')
    # of the hexagon it belongs to.
    # Store the count AFTER deduplication
    count_after = len(joined_gdf)
    # Calculate and print the number of duplicates removed
    duplicates_removed = count_before - count_after
    print(f"\nIdentified and removed {duplicates_removed} duplicate assignments.")
    print(f"Found {len(joined_gdf)} RWI points located within the hexagon grid.")
    print(joined_gdf.columns)
    # --- 6. AGGREGATE RWI VALUES PER HEXAGON ---
    # If multiple RWI points fall into a single hexagon, we need to aggregate them.
    # We will group by the hexagon's index and calculate the mean RWI.
    # We also calculate standard deviation and count for more detailed analysis.
    if not joined_gdf.empty:
        print("Aggregating RWI values for each hexagon...")
        # Group by the index of the hexagon grid ('index_right' is added by sjoin)
        rwi_agg = joined_gdf.groupby('index_right').agg(
            rwi_mean=('rwi', 'mean'),
            rwi_std=('rwi', 'std'),
            rwi_count=('rwi', 'count')
        )
    else:
        print("No RWI points were found within any hexagons. Cannot aggregate.")
        rwi_agg = pd.DataFrame(columns=['rwi_mean', 'rwi_std', 'rwi_count'])

    print("Done")

    # --- 7. MERGE AGGREGATED DATA BACK TO THE HEXAGON GRID ---
    # We now join our aggregated RWI statistics back to the original hexagon GeoDataFrame.
    print("Merging aggregated data back to the hexagon grid...")
    # Use a 'left' join to keep all original hexagons, even if they have no RWI data.
    # Hexagons with no points will have NaN values for the new columns.
    hex_grid_with_rwi = hex_grid.merge(
        rwi_agg,
        left_index=True,
        right_index=True,
        how="left"
    )

    print("Final GeoDataFrame created.")

    # --- 8. (OPTIONAL) VISUALIZE THE RESULT ---
    print("Generating map...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    hex_grid_with_rwi.plot(
        column='rwi_mean',      # Color hexagons by the mean RWI
        ax=ax,
        legend=True,
        legend_kwds={'label': "Mean Relative Wealth Index (RWI)",
                     'orientation': "horizontal"},
        missing_kwds={           # Style for hexagons with no data
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data",
        }
    )
    ax.set_title('Mean RWI per Hexagon', fontdict={'fontsize': '16', 'fontweight': '3'})
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

    investigation=False
    if investigation == True:
        print("\n--- Investigating Unjoined Points ---")

        # First, let's find the points that were NOT joined.
        # We can do this by finding the indices from the original rwi_gdf that are NOT in the joined_gdf.
        # NOTE: The 'joined_gdf' contains the original index from 'rwi_gdf' in its own index.
        joined_indices = joined_gdf.index
        unjoined_points_gdf = rwi_gdf[~rwi_gdf.index.isin(joined_indices)]

        print(f"Found {len(unjoined_points_gdf)} unjoined points, which matches our calculation (35547 - 35069 = 478).")


        # Now, let's visualize them to see where they are.
        if not unjoined_points_gdf.empty:
            print("Creating a diagnostic map to show unjoined points...")
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))

            # 1. Plot the hexagon grid as the base layer
            hex_grid.plot(
                ax=ax,
                color='lightgray',
                edgecolor='white',
                linewidth=0.5
            )

            # 2. Plot the successfully joined points in blue (optional, can be slow if many points)
            # joined_gdf.plot(
            #     ax=ax,
            #     color='blue',
            #     markersize=1,
            #     label='Joined Points'
            # )

            # 3. Plot the UNJOINED points in a highly visible color, like red.
            unjoined_points_gdf.plot(
                ax=ax,
                color='red',
                markersize=10, # Make them bigger so they are easy to see
                label='Unjoined Points'
            )

            ax.set_title('Diagnostic Map: Unjoined RWI Points (in Red)', fontsize=16)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.legend()
            plt.show()

        else:
            print("No unjoined points were found.")

    return hex_grid_with_rwi



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
    res_elec_tj_value = eb.loc[
        (eb['COMMODITY'] == app_config.UN_ELEC_CODE) &
        (eb['TRANSACTION'] == app_config.UN_HH_TRANSACTION_CODE) &
        (eb['TIME_PERIOD'] == app_config.UN_ENERGY_YEAR),
        'OBS_VALUE'
    ]
    if res_elec_tj_value.empty:
        raise ValueError(f"No UN residential energy data found for year {app_config.UN_ENERGY_YEAR} with specified codes.")
    # Check if the Series contains strings.
    if pd.api.types.is_string_dtype(res_elec_tj_value):
        # If it's a string type, remove commas from the entire series
        res_elec_tj_value = res_elec_tj_value.str.replace(',', '').iloc[0]
    else:
        res_elec_tj_value = res_elec_tj_value.iloc[0]
    res_elec_tj_value = pd.to_numeric(res_elec_tj_value)
    total_residential_elec_GWh = res_elec_tj_value / 3.6
    print(f"Total Residential electricity (UN Stats): {total_residential_elec_GWh:.0f} GWh")

    # Services electricity
    ser_elec_tj_value = eb.loc[
        (eb['COMMODITY'] == app_config.UN_ELEC_CODE) &
        (eb['TRANSACTION'] == app_config.UN_SERVICES_TRANSACTION_CODE) &
        (eb['TIME_PERIOD'] == app_config.UN_ENERGY_YEAR),
        'OBS_VALUE'
    ]
    if ser_elec_tj_value.empty:
        raise ValueError(f"No UN services energy data found for transaction {app_config.UN_SERVICES_TRANSACTION_CODE}.")
    if pd.api.types.is_string_dtype(ser_elec_tj_value):
        # If it's a string type, remove commas from the entire series
        ser_elec_tj_value = ser_elec_tj_value.str.replace(',', '').iloc[0]
    else:
        ser_elec_tj_value = ser_elec_tj_value.iloc[0]
    ser_elec_tj_value = pd.to_numeric(ser_elec_tj_value)
    total_services_elec_GWh = ser_elec_tj_value / 3.6
    print(f"Total Services electricity (UN Stats): {total_services_elec_GWh:.0f} GWh")

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
        data_HH = df_censusdata[['region', 'HH_urban', 'HH_rural','size_HH_urban', 'size_HH_rural']]
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
    df_censusdata.set_index('region', inplace=True)
    # return df_censusdata, data_HH
    return data_HH, df_censusdata