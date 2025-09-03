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
    grid_gdf["ElecPerHH_kWh_meth2"] = grid_gdf[app_config.COL_RES_ELEC_PER_HH_KWH_DHS]

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


def compare_access_to_falchetta(grid_gdf, app_config):
    """
    Compares calculated residential energy consumption tiers with Falchetta dataset tiers.

    This function bins calculated per-household energy into tiers and compares the
    distribution of households across these tiers against pre-loaded Falchetta tier data.
    It also performs a similarity analysis between the DHS-based calculated tiers and
    Falchetta's majority tier.

    Args:
        grid_gdf: GeoDataFrame of the hexagonal grid with energy consumption data.
        app_config: The configuration module.

    Returns:
        GeoDataFrame: grid_gdf, potentially with added columns for tiering/comparison.
    """
    print("Comparing access tiers to Falchetta dataset...")

    def calculate_tier_share_method(data_grid, method_suffix, hh_with_access_col, hh_wo_access_col, category_total_val):
        # Helper for tier share calculation
        tier_col_name = f'tiers_{method_suffix}'
        if tier_col_name not in data_grid.columns:
            # print(f"Warning: Tier column '{tier_col_name}' not found for method '{method_suffix}'.")
            return pd.Series(dtype=float)
        if category_total_val == 0: return pd.Series(dtype=float)

        tier_share = data_grid.groupby(tier_col_name)[hh_with_access_col].sum()
        if 0 in tier_share.index :
            tier_share.loc[0] += data_grid[hh_wo_access_col].sum()
        else:
            tier_share.loc[0] = data_grid[hh_wo_access_col].sum()
        return tier_share.sort_index() / category_total_val

    bins_tiers = app_config.BINS_TIERS_ENERGY
    tier_labels = range(len(bins_tiers) - 1)

    categories_summary = {
        'national': app_config.COL_HH_TOTAL, 'urban': app_config.COL_HH_URBAN, 'rural': app_config.COL_HH_RURAL
    }

    # Falchetta dataset
    for col_type in [app_config.COL_TIERS_FALCHETTA_MAJ, app_config.COL_TIERS_FALCHETTA_MEAN]:
        if col_type in grid_gdf.columns:
            tiers_summary_df = pd.DataFrame()
            for cat_name, total_hh_col in categories_summary.items():
                 if total_hh_col in grid_gdf.columns and grid_gdf[total_hh_col].sum() > 0:
                    cat_sum = grid_gdf.groupby(col_type)[total_hh_col].sum()
                    tiers_summary_df[cat_name] = cat_sum / cat_sum.sum()
            print(f"\nFalchetta Tiers Summary ({col_type}):")
            print(tiers_summary_df.fillna(0))

    # Our methods
    methods_to_compare = {
        'meth1': app_config.COL_RES_ELEC_PER_HH_LOG,
        'meth2': app_config.COL_RES_ELEC_PER_HH_KWH_DHS
    }
    categories_for_comparison = [
        ('national', app_config.COL_HH_WITH_ACCESS, app_config.COL_HH_WO_ACCESS, app_config.COL_HH_TOTAL),
        ('urban', app_config.COL_HH_WITH_ACCESS_URB, app_config.COL_HH_WO_ACCESS_URB, app_config.COL_HH_URBAN),
        ('rural', app_config.COL_HH_WITH_ACCESS_RUR, app_config.COL_HH_WO_ACCESS_RUR, app_config.COL_HH_RURAL)
    ]

    for method_key, energy_col_name in methods_to_compare.items():
        if energy_col_name not in grid_gdf.columns:
            print(f"Warning: Energy column '{energy_col_name}' for method '{method_key}' not found.")
            continue

        grid_gdf[f'tiers_{method_key}'] = pd.cut(grid_gdf[energy_col_name], bins=bins_tiers, labels=tier_labels, right=False)
        grid_gdf[f'tiers_{method_key}'] = grid_gdf[f'tiers_{method_key}'].fillna(0).astype(int)

        df_tiers_data = pd.DataFrame()
        for cat_name, hh_access_col, hh_no_access_col, total_hh_col in categories_for_comparison:
            if all(c in grid_gdf.columns for c in [hh_access_col, hh_no_access_col, total_hh_col]):
                cat_total_val = grid_gdf[total_hh_col].sum()
                if cat_total_val > 0:
                    tier_share_series = calculate_tier_share_method(grid_gdf, method_key, hh_access_col, hh_no_access_col, cat_total_val)
                    df_tiers_data[cat_name] = tier_share_series

        print(f"\nTier Shares for Method '{method_key}':")
        print(df_tiers_data.fillna(0))

    if f'tiers_meth2' in grid_gdf.columns and app_config.COL_TIERS_FALCHETTA_MAJ in grid_gdf.columns:
        grid_gdf['tiers_DHS_adjusted'] = grid_gdf['tiers_meth2'].where(grid_gdf['tiers_meth2'] != 5, 4)
        grid_gdf['Similarity_Falchetta_DHS'] = grid_gdf['tiers_DHS_adjusted'] == grid_gdf[app_config.COL_TIERS_FALCHETTA_MAJ]
        grid_gdf['Difference_Falchetta_DHS'] = abs(pd.to_numeric(grid_gdf['tiers_DHS_adjusted']) - pd.to_numeric(grid_gdf[app_config.COL_TIERS_FALCHETTA_MAJ]))

        print("\nSimilarity Analysis (Falchetta vs DHS-Method2):")
        print(f"Number of lines with similar tiers: {grid_gdf['Similarity_Falchetta_DHS'].sum()}")
        print(f"Mean difference in tiers: {grid_gdf['Difference_Falchetta_DHS'].mean():.2f}")
        print(f"Median difference in tiers: {grid_gdf['Difference_Falchetta_DHS'].median():.2f}")

    print("Finished Falchetta comparison.")
    return grid_gdf