# -*- coding: utf-8 -*-
"""
Module for loading data related to the Industry energy demand analysis.
"""
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
    admin_gpkg_path = app_config.ADMIN_PATH / app_config.ADMIN_GPKG
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
    hexagons_path = app_config.OUTPUT_DIR / app_config.H3_GRID_HEX_SHP
    hexagons = gpd.read_file(hexagons_path)
    grid_gdf = hexagons
    print(f"Hexagon grid loaded: {grid_gdf.shape}")

    return regions_list, admin_gdf, region_gdf, grid_gdf, hexagons

def load_and_process_energy_balance(app_config):
    """Loads and processes energy balance data from UN stats."""
    print("Loading and processing energy balance data...")
    eb = pd.read_csv(app_config.ENERGY_BALANCE_PATH / app_config.UN_ENERGY_BALANCE_CSV)
    
    code_elec = app_config.UN_ELEC_CODE
    code_oil = app_config.UN_OIL_CODE
    code_ind_nFM = app_config.UN_INDUSTRY_NFM
    code_ind_mining = app_config.UN_INDUSTRY_MINING
    year = app_config.UN_ENERGY_YEAR

    def get_energy_value(df, commodity_code, transaction_code):
        val = df.loc[
            (df['COMMODITY'] == commodity_code) & 
            (df['TRANSACTION'] == transaction_code) & 
            (df['TIME_PERIOD'] == year), 
            'OBS_VALUE'
        ]
        if not val.empty:
            return pd.to_numeric(val.str.replace(',', '').iloc[0])
        print(f"Warning: No energy value found for {commodity_code}, {transaction_code}, {year}. Returning 0.")
        return 0 # Return 0 if no value found to avoid errors

    energy_data = {
        'elec_nonFerrousMetals_TJ': get_energy_value(eb, code_elec, code_ind_nFM),
        'elec_mining_TJ': get_energy_value(eb, code_elec, code_ind_mining),
        'oil_nonFerrousMetals_TJ': get_energy_value(eb, code_oil, code_ind_nFM),
        'oil_mining_TJ': get_energy_value(eb, code_oil, code_ind_mining)
    }
    
    energy_data['elec_ind_TJ'] = energy_data['elec_nonFerrousMetals_TJ'] + energy_data['elec_mining_TJ']
    energy_data['oil_ind_TJ'] = energy_data['oil_nonFerrousMetals_TJ'] + energy_data['oil_mining_TJ']
    energy_data['energy_ind_TJ'] = energy_data['elec_ind_TJ'] + energy_data['oil_ind_TJ']
    
    print("Energy balance data processed. UN stats:")
    for key, value in energy_data.items():
        print(f"{key}: {value:,.1f} TJ")
        
    return energy_data

def load_raw_mines_csv(app_config):
    """Loads the raw mines input CSV file."""
    print("Loading raw mines CSV data...")
    mines_input_df = pd.read_csv(app_config.MINES_INPUT_CSV)
    print("Raw mines CSV data loaded.")
    return mines_input_df
