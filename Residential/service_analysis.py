import pandas as pd


def calculate_service_buildings_based_elec(grid_gdf, app_config, total_services_elec_gwh):
    """
    Calculates services electricity demand based on the number of accessible service buildings.

    It estimates the number of service buildings, then those with access, and
    distributes the total national services electricity amongst them.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_elec_gwh: Total national services energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for building-based services energy.
    """
    print("Calculating services electricity (building-based)...")

    if not all(col in grid_gdf.columns for col in [app_config.COL_BUILDINGS_SUM, app_config.COL_RES_BUI, app_config.PROB_ELEC_COL]):
        raise KeyError("One or more required columns for service building electricity calculation are missing.")

    grid_gdf[app_config.COL_SER_BUI] = grid_gdf[app_config.COL_BUILDINGS_SUM] - grid_gdf[app_config.COL_RES_BUI]
    grid_gdf[app_config.COL_SER_BUI_ACC] = grid_gdf[app_config.COL_SER_BUI] * grid_gdf[app_config.PROB_ELEC_COL]

    total_ser_bui_with_access = grid_gdf[app_config.COL_SER_BUI_ACC].sum()
    print(f"Total services buildings with estimated access: {total_ser_bui_with_access:,.0f}")

    ser_elec_per_bui_kwh = (total_services_elec_gwh * 1e6) / total_ser_bui_with_access if total_ser_bui_with_access > 0 else 0
    if total_ser_bui_with_access == 0: print("Warning: Total service buildings with access is 0.")

    print(f"Service electricity per accessible building: {ser_elec_per_bui_kwh:,.0f} kWh/building")
    grid_gdf[app_config.COL_SER_ELEC_KWH_BUI] = ser_elec_per_bui_kwh * grid_gdf[app_config.COL_SER_BUI_ACC]

    print("Finished calculating services electricity (building-based).")
    return grid_gdf


def calculate_service_gdp_based_elec(grid_gdf, app_config, total_services_elec_gwh):
    """
    Calculates services energy demand based on GDP data.

    If GDP data is available, it
    distributes total national services energy based on the GDP of each grid cell.
    Otherwise, it sets the GDP-based energy column to zero.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_energy_gwh: Total national services energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for GDP-based services energy.
    """
    print("Calculating services energy (GDP-based)...")

    gdp_col = app_config.COL_GDP_PPP_MEAN
    col_ser_en_gdp = app_config.COL_SER_ELEC_KWH_GDP

    if gdp_col and gdp_col in grid_gdf.columns:
        total_gdp_kdollars = grid_gdf[gdp_col].sum() / 1000
        # print(f"Total GDP: {total_gdp_kdollars:,.0f} k$")
        ser_elec_per_gdp_kwh_per_kdolar = (total_services_elec_gwh * 1e6) / total_gdp_kdollars if total_gdp_kdollars > 0 else 0
        if total_gdp_kdollars == 0: print("Warning: Total GDP is 0.")

        print(f"Service energy per unit of GDP: {ser_elec_per_gdp_kwh_per_kdolar:,.2f} kWh/k$")
        grid_gdf[col_ser_en_gdp] = ser_elec_per_gdp_kwh_per_kdolar * (grid_gdf[gdp_col] / 1000)
        print(f"'{col_ser_en_gdp}' column created/updated.")
    else:
        grid_gdf[col_ser_en_gdp] = 0.0
        print(f"Warning: GDP column '{gdp_col}' not found or not defined. GDP-based service energy set to 0.")

    print("Finished calculating services energy (GDP-based).")
    return grid_gdf


def _load_dhs_employee_data(app_config):
    """ Loads DHS employee and working population share data. """
    path_emp_women = app_config.DHS_EMPLOYEE_WOMEN_CSV
    path_emp_men = app_config.DHS_EMPLOYEE_MEN_CSV
    path_work_pop = app_config.DHS_WORKING_POP_SHARE_CSV

    data_employee_women = pd.read_csv(path_emp_women, index_col=(0, 1)) # province, location_status
    data_employee_men = pd.read_csv(path_emp_men, index_col=(0, 1))   # province, location_status
    data_workingpop_share = pd.read_csv(path_work_pop, index_col=(1, 0)) # location_status, sex

    # Sum employee shares for relevant categories from config
    emp_categories = app_config.DHS_EMPLOYMENT_CATEGORIES
    data_employee_women['total_employee_share_women'] = data_employee_women[emp_categories].sum(axis=1)
    data_employee_men['total_employee_share_men'] = data_employee_men[emp_categories].sum(axis=1)

    return data_employee_women, data_employee_men, data_workingpop_share

def calculate_employee_based_electricity(grid_gdf, app_config, total_services_elec_gwh, df_censusdata):
    """
    Calculates services energy demand based on the estimated number of employees.

    This function uses DHS survey data for employment rates and working population shares,
    combined with census data for population distribution, to estimate the number of
    employees per grid cell. Total national services energy is then distributed
    among employees with electricity access.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_services_elec_gwh: Total national services energy (GWh) from UN stats.
        df_censusdata: DataFrame with provincial census data, indexed by region name.

    Returns:
        GeoDataFrame: grid_gdf with added columns for employee counts and employee-based services energy.
    """
    print("Calculating services energy (employee-based)...")

    # Load data
    try:
        data_employee_women, data_employee_men, data_workingpop_share = _load_dhs_employee_data(app_config)
    except FileNotFoundError as e:
        print(f"Error: Employee data file not found: {e}. Skipping employee-based service energy calculation.")
        grid_gdf[app_config.COL_SER_ELEC_KWH_EMP] = 0.0
        return grid_gdf

    # Ensure required columns exist in census data
    required_census_cols = ['Share women', 'size_HH_urban', 'size_HH_rural']
    if not all(col in df_censusdata.columns for col in required_census_cols):
        raise KeyError(f"Missing one or more required columns in census data: {required_census_cols}")

    # --- FIX START: The calculation logic is moved into a more robust helper function ---

    # Make a local copy to avoid SettingWithCopyWarning
    df_censusdata_local = df_censusdata.copy()

    # Define a single, more robust helper function for calculating population
    def calculate_nb_gender(row, gender_type):
        admin_name = row[app_config.COL_ADMIN_NAME]
        loc_status = row[app_config.COL_LOC_ASSESSED]
        hh_total = row[app_config.COL_HH_TOTAL]

        # Get HH size for the specific location type (urban/rural)
        hh_size = df_censusdata_local.loc[admin_name, f"size_HH_{loc_status}"]

        # Determine sex share and working pop share based on gender_type
        if gender_type == 'women':
            regional_sex_share = df_censusdata_local.loc[admin_name, 'Share women']
            working_age_pop_share = data_workingpop_share.loc[('Female', loc_status), '15-49'] / 100
        elif gender_type == 'men':
            regional_sex_share = 1 - df_censusdata_local.loc[admin_name, 'Share women']
            working_age_pop_share = data_workingpop_share.loc[('Male', loc_status), '15-49'] / 100
        else:
            raise ValueError('Unknown gender_type', gender_type)

        return hh_total * hh_size * regional_sex_share * working_age_pop_share

    # Apply the corrected function for both women and men
    print("  Calculating number of men and women (15-49)...")
    grid_gdf['nb_women'] = grid_gdf.apply(calculate_nb_gender, args=('women',), axis=1)
    grid_gdf['nb_men'] = grid_gdf.apply(calculate_nb_gender, args=('men',), axis=1)

    # --- FIX END ---

    # Calculate working women/men (This part was mostly correct)
    def calculate_working_gender(row, sex_col_name, employee_data_df, employee_share_col_name):
        loc_status = row[app_config.COL_LOC_ASSESSED]
        # Normalize the region name to match the index in the employee data
        admin_name_processed = row[app_config.COL_ADMIN_NAME].lower().replace('-', ' ')

        try:
            # Look up the working share from the pre-loaded employee data
            percent_working = employee_data_df.loc[(admin_name_processed, loc_status), employee_share_col_name] / 100
        except KeyError:
            # If a specific region/location combo is missing, default to 0 to avoid errors
            percent_working = 0

        return row[sex_col_name] * percent_working

    print("  Calculating number of working men and women...")
    grid_gdf['nb_women_working'] = grid_gdf.apply(calculate_working_gender, args=('nb_women', data_employee_women, 'total_employee_share_women'), axis=1)
    grid_gdf['nb_men_working'] = grid_gdf.apply(calculate_working_gender, args=('nb_men', data_employee_men, 'total_employee_share_men'), axis=1)

    # Sum up totals
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE] = grid_gdf['nb_women_working'] + grid_gdf['nb_men_working']
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS] = grid_gdf.loc[grid_gdf[app_config.COL_STATUS_ELECTRIFIED] == 'elec', app_config.COL_TOTAL_EMPLOYEE]
    grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS].fillna(0, inplace=True) # Ensure non-elec rows are 0, not NaN

    total_employee_national_with_access = grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS].sum()
    print(f"Total employees with access: {total_employee_national_with_access:,.0f}")

    if total_employee_national_with_access > 0:
        ser_en_per_employee_kwh = (total_services_elec_gwh * 1e6) / total_employee_national_with_access # kWh / employee
    else:
        print("Warning: Total employees with access is 0. Energy per employee will be 0.")
        ser_en_per_employee_kwh = 0

    print(f"Service electricity per accessible employee: {ser_en_per_employee_kwh:,.2f} kWh/employee")

    # Distribute energy based on employees with access
    grid_gdf[app_config.COL_SER_ELEC_KWH_EMP] = ser_en_per_employee_kwh * grid_gdf[app_config.COL_TOTAL_EMPLOYEE_WITH_ACCESS]

    print("Finished calculating services energy (employee-based).")
    return grid_gdf