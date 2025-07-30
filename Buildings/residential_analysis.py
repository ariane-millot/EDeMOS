from scipy.optimize import fsolve
import numpy as np
import pandas as pd


def calculate_energy_per_hh_method1(grid_gdf, app_config, total_residential_elec_GWh):
    """
    Calculates energy per household using RWI-based logistic function

    This method normalizes the Relative Wealth Index (RWI), then solves for a
    parameter `k` in a logistic function to ensure the total calculated energy
    matches UN statistics. It adds a column for the calculated energy per household.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid.
        app_config: The configuration module.
        total_residential_energy_gwh: Total national residential energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added column for energy per household (Method 1).
    """
    print("Calculating energy per HH (Method 1: RWI-logistic)...")

    if app_config.COL_RWI_MEAN not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_RWI_MEAN}' not found for RWI normalization.")

    rwi_min = grid_gdf[app_config.COL_RWI_MEAN].min()
    rwi_max = grid_gdf[app_config.COL_RWI_MEAN].max()
    if (rwi_max - rwi_min) == 0:
        grid_gdf[app_config.COL_RWI_NORM] = 0.5
        print("Warning: RWI min and max are equal. Normalized RWI set to 0.5.")
    else:
        grid_gdf[app_config.COL_RWI_NORM] = (grid_gdf[app_config.COL_RWI_MEAN] - rwi_min) / (rwi_max - rwi_min)

    alpha = app_config.LOGISTIC_E_THRESHOLD / app_config.LOGISTIC_ALPHA_DERIVATION_THRESHOLD - 1

    if app_config.COL_HH_WITH_ACCESS not in grid_gdf.columns:
        raise KeyError(f"Required column '{app_config.COL_HH_WITH_ACCESS}' not found for fsolve.")

    def func_solve_k(k_var):
        # Calculates total energy based on k_var and compares to UN total.
        e_hh = app_config.LOGISTIC_E_THRESHOLD / (1 + alpha * np.exp(-k_var * grid_gdf[app_config.COL_RWI_NORM]))
        res_energy_assessed = (e_hh * grid_gdf[app_config.COL_HH_WITH_ACCESS]).sum()
        return res_energy_assessed / 1e6 - total_residential_elec_GWh # kWh to GWh

    try:
        k_solution = fsolve(func_solve_k, app_config.LOGISTIC_K_INITIAL_GUESS)
        print(f"Solved k for logistic function: {k_solution[0]:.4f}")
        k_to_use = k_solution[0]
    except Exception as e:
        print(f"Error solving for k in RWI-logistic method: {e}. Using initial guess: {app_config.LOGISTIC_K_INITIAL_GUESS}")
        k_to_use = app_config.LOGISTIC_K_INITIAL_GUESS

    grid_gdf[app_config.COL_RES_ELEC_PER_HH_LOG] = app_config.LOGISTIC_E_THRESHOLD / (
        1 + alpha * np.exp(-k_to_use * grid_gdf[app_config.COL_RWI_NORM])
    )
    print("Finished calculating energy per HH (Method 1).")
    return grid_gdf, k_to_use


def calculate_total_residential_electricity(grid_gdf, app_config, total_residential_energy_gwh):
    """
    Calculates total residential energy per cell for different methods and scales to UN stats.

    This function takes the per-household energy estimates from Method 1 (RWI-logistic)
    and Method 2 (DHS-based), calculates the total energy per grid cell for each method,
    and then scales these totals so that the national aggregate matches UN statistics.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid with per-HH energy estimates.
        app_config: The configuration module.
        total_residential_energy_gwh: Total national residential energy (GWh) from UN stats.

    Returns:
        GeoDataFrame: grid_gdf with added columns for raw and scaled total residential energy.
    """
    print("Calculating total residential energy and scaling...")

    # Ensure required input columns exist
    required_cols_meth1 = [app_config.COL_RES_ELEC_PER_HH_LOG, app_config.COL_HH_WITH_ACCESS]
    required_cols_meth2 = [app_config.COL_RES_ELEC_PER_HH_KWH_DHS, app_config.COL_HH_WITH_ACCESS]

    for col in required_cols_meth1:
        if col not in grid_gdf.columns: raise KeyError(f"Method 1: Column '{col}' not found.")
    for col in required_cols_meth2:
        if col not in grid_gdf.columns: raise KeyError(f"Method 2: Column '{col}' not found.")

    # For method 1, assign electricity per HH where access > 0, else 0
    grid_gdf["ElecPerHH_kWh_meth1"] = grid_gdf.apply(
        lambda row: row[app_config.COL_RES_ELEC_PER_HH_LOG] if row[app_config.COL_HH_WITH_ACCESS] > 0 else 0, axis=1
    )
    # For method 2, the ElectricityPerHH_DHS is already calculated considering access within its script.
    grid_gdf["ElecPerHH_kWh_meth2"] = grid_gdf[app_config.COL_RES_ELEC_PER_HH_DHS]

    methods_map = {
        'meth1': {'per_hh_col': "ElecPerHH_kWh_meth1", 'output_col_scaled': app_config.COL_RES_ELEC_KWH_METH1_SCALED, 'raw_total_col': app_config.COL_RES_ELEC_KWH_METH1},
        'meth2': {'per_hh_col': "ElecPerHH_kWh_meth2", 'output_col_scaled': app_config.COL_RES_ELEC_KWH_METH2_SCALED, 'raw_total_col': app_config.COL_RES_ELEC_KWH_METH2}
    }

    results_beforescaling_summary = {}
    results_afterscaling_summary = {}

    for method_key, details in methods_map.items():
        # Calculate raw total energy per cell (kWh)
        grid_gdf[details['raw_total_col']] = grid_gdf[app_config.COL_HH_WITH_ACCESS] * grid_gdf[details['per_hh_col']]

        # Aggregate by administrative region (e.g., NAME_1) if COL_ADMIN_NAME is present
        if app_config.COL_ADMIN_NAME in grid_gdf.columns:
            regional_sum_gwh = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[details['raw_total_col']].sum() / 10**6 # kWh to GWh
            results_beforescaling_summary[method_key] = regional_sum_gwh
            total_assessed_gwh = regional_sum_gwh.sum()
        else: # No admin column, sum all cells
            total_assessed_gwh = grid_gdf[details['raw_total_col']].sum() / 10**6 # kWh to GWh
            results_beforescaling_summary[method_key] = pd.Series({"National": total_assessed_gwh})


        if total_assessed_gwh == 0:
            print(f"Warning: Total assessed energy for {method_key} is 0. Scaling factor cannot be computed. Scaled energy will be 0.")
            scaling_factor = 0
        else:
            scaling_factor = total_residential_energy_gwh / total_assessed_gwh

        print(f"Method {method_key}: Total Assessed = {total_assessed_gwh:.2f} GWh, UN Stats = {total_residential_energy_gwh:.2f} GWh, Scaling Factor = {scaling_factor:.4f}")

        grid_gdf[details['output_col_scaled']] = grid_gdf[details['raw_total_col']] * scaling_factor

        if app_config.COL_ADMIN_NAME in grid_gdf.columns:
            results_afterscaling_summary[method_key] = grid_gdf.groupby(app_config.COL_ADMIN_NAME)[details['output_col_scaled']].sum() / 10**6
        else:
            results_afterscaling_summary[method_key] = pd.Series({"National": grid_gdf[details['output_col_scaled']].sum() / 10**6})

    print("\nSummary of energy consumption before scaling (GWh):")
    print(pd.DataFrame(results_beforescaling_summary))
    print("\nSummary of energy consumption after scaling (GWh):")
    print(pd.DataFrame(results_afterscaling_summary))

    print("Finished calculating and scaling total residential energy.")
    return grid_gdf