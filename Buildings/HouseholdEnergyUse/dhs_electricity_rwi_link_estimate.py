"""
DHS electricity rwi link estimate
This script estimates residential electricity consumption by linking DHS survey data
with a high-resolution Relative Wealth Index (RWI) grid.

The workflow is as follows:
1. Load and prepare DHS household survey data.
2. Aggregate household data to the DHS cluster level, calculating metrics like
   average wealth, access rate, and average consumption for connected households.
3. Load grid-level data (RWI, population, etc.).
4. Calibrate the grid RWI to match the distribution of the DHS wealth index on a
   per-region basis using a weighted quantile mapping technique.
5. Train separate k-Nearest Neighbors (k-NN) models for urban and rural areas to
   predict average household electricity consumption based on the calibrated RWI
   and electricity access rate.
6. Apply the models to the grid data to predict consumption for each grid cell.
7. Calculate the total estimated electricity consumption.
"""

import sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# --- Constants for Column Names ---
# DHS Derived Columns
WI_COMBINED = 'wi_combined'
WEIGHT = 'weight'
HAS_ACCESS = 'has_access'
AVG_WI = 'avg_wi'
ACCESS_RATE = 'access_rate'
NUM_HH = 'num_hh'
AVG_ELEC_CONNECTED = 'avg_elec_connected'


def weighted_quantile_map(source_series, source_weights, target_series, target_weights):
    """
    Maps values from a source distribution to a target distribution using
    weighted quantiles.

    Args:
        source_series: The series of values to be mapped (e.g., grid RWI).
        source_weights: The weights for the source series (e.g., household counts).
        target_series: The series with the target distribution (e.g., cluster WI).
        target_weights: The weights for the target series (e.g., cluster weights).

    Returns:
        A pandas Series containing the mapped values, with the same index as source_series.
    """
    # --- Create a weighted CDF for the source data ---
    source_df = pd.DataFrame({'val': source_series, 'w': source_weights}).dropna()
    source_df = source_df.sort_values('val')
    source_df['cdf'] = np.cumsum(source_df['w']) / np.sum(source_df['w'])

    # --- Create a weighted CDF for the target data ---
    target_df = pd.DataFrame({'val': target_series, 'w': target_weights}).dropna()
    target_df = target_df.sort_values('val')
    target_df['cdf'] = np.cumsum(target_df['w']) / np.sum(target_df['w'])

    # --- Map the source to the target via the CDFs ---
    source_percentiles = np.interp(source_series, source_df['val'], source_df['cdf'])
    mapped_values = np.interp(source_percentiles, target_df['cdf'], target_df['val'])

    return pd.Series(mapped_values, index=source_series.index)


def aggregate_dhs_to_clusters(dhs_data: pd.DataFrame, app_config) -> pd.DataFrame:
    """Aggregates household-level DHS data to the cluster level."""
    print("\n--- 2. Aggregating DHS Data to Cluster Level ---")

    # Aggregation on ALL households to get cluster features
    agg_all_hh = {
        AVG_WI: (WI_COMBINED, 'mean'),
        ACCESS_RATE: (HAS_ACCESS, 'mean'),
        NUM_HH: (app_config.DHS_CLUSTER, 'size'),
        app_config.DHS_PROVINCE: (app_config.DHS_PROVINCE, 'first'),
        app_config.DHS_URBAN_RURAL: (app_config.DHS_URBAN_RURAL, 'first')
    }
    clusters_features_df = dhs_data.groupby(app_config.DHS_CLUSTER).agg(**agg_all_hh)

    # Aggregation on CONNECTED households to get the target variable
    dhs_with_access = dhs_data[dhs_data[HAS_ACCESS] == 1].copy()
    agg_connected_hh = {
        AVG_ELEC_CONNECTED: (app_config.DHS_ELEC_KWH_ASSESSED_SURVEY, 'mean')
    }
    clusters_target_df = dhs_with_access.groupby(app_config.DHS_CLUSTER).agg(**agg_connected_hh)

    # Merge features and target, keeping all clusters
    final_clusters_df = pd.merge(
        clusters_features_df,
        clusters_target_df,
        on=app_config.DHS_CLUSTER,
        how='left'
    ).reset_index()

    # Fill NaN for avg consumption with 0 for clusters with no access
    final_clusters_df[AVG_ELEC_CONNECTED] = final_clusters_df[AVG_ELEC_CONNECTED].fillna(0)

    print(f"Aggregated into {len(final_clusters_df)} unique clusters.")
    return final_clusters_df

def calibrate_rwi_by_region(grid_df: pd.DataFrame, clusters_df: pd.DataFrame, app_config) -> pd.DataFrame:
    """Calibrates grid RWI against DHS cluster WI for each administrative region."""
    print("\n--- 3. Calibrating Grid RWI by Region ---")

    grid_df = grid_df.copy()
    grid_df[app_config.COL_RWI_REGION_MODIFIED] = np.nan

    # --- Step 1: Normalize Region/Province Names for Robust Matching ---
    # Define a reusable normalization function
    def normalize_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        return name.lower().replace('-', '').replace(' ', '')

    # Apply normalization to create temporary matching keys in both dataframes
    grid_df['_norm_region'] = grid_df[app_config.COL_ADMIN_NAME].apply(normalize_name)
    clusters_df['_norm_province'] = clusters_df[app_config.DHS_PROVINCE].apply(normalize_name)

    regions = grid_df['_norm_region'].unique()

    for region in regions:
        print(f"Calibrating for region: {region}")

        # Filter data for the current region
        grid_region = grid_df[grid_df['_norm_region'] == region].copy()
        clusters_region = clusters_df[clusters_df['_norm_province'] == region]
        if clusters_region.empty:
            print(f"  - WARNING: No DHS cluster data for region '{region}'. Skipping calibration for this region.")
            continue

        # Prepare inputs for the calibration function for the region
        grid_series_region = grid_region[app_config.COL_RWI_MEAN]
        grid_weights_region = grid_region[app_config.COL_HH_TOTAL]
        target_series_region = clusters_region[AVG_WI]
        target_weights_region = pd.Series(np.ones(len(clusters_region)), index=clusters_region.index) # uniform weights

        # Apply mapping and update the main grid dataframe
        calibrated_rwi = weighted_quantile_map(
            grid_series_region, grid_weights_region, target_series_region, target_weights_region
        )
        grid_df.loc[grid_df['_norm_region'] == region, app_config.COL_RWI_REGION_MODIFIED] = calibrated_rwi

    # Handle regions that were not calibrated (if any) by falling back to original RWI
    uncalibrated_mask = grid_df[app_config.COL_RWI_REGION_MODIFIED].isnull()
    if uncalibrated_mask.any():
        print(f"Warning: {uncalibrated_mask.sum()} grid cells could not be calibrated. Falling back to original RWI.")
        grid_df.loc[uncalibrated_mask, app_config.COL_RWI_REGION_MODIFIED] = grid_df.loc[uncalibrated_mask, app_config.COL_RWI_MEAN]

    return grid_df


def plot_rwi_distributions(grid_df: pd.DataFrame, clusters_df: pd.DataFrame, app_config):
    """Plots the distributions of original, calibrated, and target RWI."""
    print("\n--- Plotting RWI Distributions for Comparison ---")
    plt.figure(figsize=(12, 6))

    # Calculate household weights for plotting
    grid_df['pct_households'] = grid_df[app_config.COL_HH_TOTAL] / grid_df[app_config.COL_HH_TOTAL].sum()

    sns.kdeplot(data=grid_df, x=app_config.COL_RWI_MEAN, weights='pct_households',
                label='Original Meta RWI (Weighted)', color='gray', lw=2)

    sns.kdeplot(data=clusters_df, x=AVG_WI,
                label='Target DHS Cluster WI', color='green', lw=3, linestyle='--')

    sns.kdeplot(data=grid_df, x=app_config.COL_RWI_REGION_MODIFIED, weights='pct_households',
                label='Region-Calibrated Grid RWI (Weighted)', color='orange', lw=3)

    plt.title('Comparison of wealth index distributions', fontsize=16)
    plt.xlabel('DHS Wealth Index / RWI')
    plt.ylabel('Density')
    plt.legend()
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def train_and_predict_consumption(grid_df: pd.DataFrame, clusters_df: pd.DataFrame, app_config, k_neighbors: int) -> pd.DataFrame:
    """Trains k-NN models for urban/rural areas and predicts HH consumption."""
    print("\n--- 4. Training Models and Predicting Consumption ---")
    grid_df = grid_df.copy()

    # 1. Prepare grid data for prediction
    # Use calibrated RWI for analysis
    grid_df[app_config.COL_RWI_ANALYSIS] = grid_df[app_config.COL_RWI_REGION_MODIFIED]
    # Calculate access rate, ensuring no division by zero
    grid_df[ACCESS_RATE] = (grid_df[app_config.COL_HH_WITH_ACCESS] / grid_df[app_config.COL_HH_TOTAL]).fillna(0)

    # 2. Split data into Urban and Rural subsets
    urban_clusters = clusters_df[clusters_df[app_config.DHS_URBAN_RURAL] == 'urban']
    rural_clusters = clusters_df[clusters_df[app_config.DHS_URBAN_RURAL] == 'rural']
    urban_grid = grid_df[grid_df[app_config.COL_LOC_ASSESSED] == 'urban'].copy()
    rural_grid = grid_df[grid_df[app_config.COL_LOC_ASSESSED] == 'rural'].copy()

    predictions = []

    # 3. Process each subset (Urban/Rural)
    for subset_name, grid_subset, cluster_subset in [('Urban', urban_grid, urban_clusters),
                                                     ('Rural', rural_grid, rural_clusters)]:
        print(f"--- Processing {subset_name} Subset ---")
        if grid_subset.empty or cluster_subset.empty:
            print(f"Skipping {subset_name} model: No data available in grid or cluster subset.")
            continue

        # Define features and target
        features = [AVG_WI, ACCESS_RATE]
        target = AVG_ELEC_CONNECTED

        X_train = cluster_subset[features]
        y_train = cluster_subset[target]

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Create and train k-NN model
        knn_model = KNeighborsRegressor(n_neighbors=k_neighbors, weights='uniform')
        knn_model.fit(X_train_scaled, y_train)

        # Prepare grid data for prediction
        # Rename grid columns to match training features
        X_predict = grid_subset[[app_config.COL_RWI_ANALYSIS, ACCESS_RATE]]
        X_predict.columns = features
        X_predict_scaled = scaler.transform(X_predict)

        # Predict and store results
        preds = knn_model.predict(X_predict_scaled)
        predictions.append(pd.Series(preds, index=grid_subset.index))

    # 4. Combine predictions back into the main grid dataframe
    if predictions:
        grid_df[app_config.COL_RES_ELEC_PER_HH_KWH_DHS] = pd.concat(predictions)
        grid_df[app_config.COL_RES_ELEC_PER_HH_KWH_DHS] = grid_df[app_config.COL_RES_ELEC_PER_HH_KWH_DHS].fillna(0)
    else:
        print("Warning: No predictions were made. Final consumption will be zero.")
        grid_df[app_config.COL_RES_ELEC_PER_HH_KWH_DHS] = 0

    return grid_df


def estimate_electricity_rwi_link(grid_gdf, app_config):

    # Run the script to assess electricity consumption of households in the DHS dataset
    recalculate_energy_perhh = app_config.DHS_RECALCULATE_ENERGY_PERHH
    if recalculate_energy_perhh:
        from Buildings.HouseholdEnergyUse.estimate_energy_perhh_DHS import compute_energy_perhh_dhs
        compute_energy_perhh_dhs(app_config)

    # --- Step 1: Load DHS Data ---
    print("--- 1. Loading and Preparing DHS Data ---")
    dhs_data = pd.read_csv(app_config.DHS_HOUSEHOLD_DATA_CSV)

    # Scale raw values and create standard columns
    dhs_data[WI_COMBINED] = 1e-5 * dhs_data[app_config.DHS_WEALTH_INDEX]
    dhs_data[WEIGHT] = 1e-6 * dhs_data[app_config.DHS_WEIGHT]
    dhs_data[HAS_ACCESS] = dhs_data[app_config.DHS_ELEC_ACCESS].astype(int)

    print(f"Loaded {len(dhs_data)} household records.")

    # --- Step 2: Aggregate DHS Data ---
    final_clusters_df = aggregate_dhs_to_clusters(dhs_data, app_config)

    # --- Step 3: Calibrate RWI in grid---
    grid_gdf = calibrate_rwi_by_region(grid_gdf, final_clusters_df, app_config)

    # Optional: Visualize the result of the calibration
    plot_rwi_distributions(grid_gdf, final_clusters_df, app_config)

    # --- Step 4: Train Models and Predict ---
    grid_with_predictions = train_and_predict_consumption(grid_gdf, final_clusters_df, app_config, k_neighbors=11)

    # --- Step 5: Final Calculation and Output ---
    grid_with_predictions[app_config.COL_RES_TOTAL_ELEC_KWH_DHS] = (
        grid_with_predictions[app_config.COL_RES_ELEC_PER_HH_KWH_DHS] * grid_with_predictions[app_config.COL_HH_WITH_ACCESS]
    )

    print("\n--- 5. Final Results ---")
    total_gwh = grid_with_predictions[app_config.COL_RES_TOTAL_ELEC_KWH_DHS].sum() / 1e6
    print(f"Prediction complete. Total predicted consumption for the country: {total_gwh:,.2f} GWh")
    # Group by the location type and sum the predicted energy for each group
    consumption_by_type = grid_with_predictions.groupby(app_config.COL_LOC_ASSESSED)[app_config.COL_RES_TOTAL_ELEC_KWH_DHS].sum()
    # Get urban and rural totals (using .get() for safety in case a category is missing)
    urban_kwh = consumption_by_type.get('urban', 0)
    rural_kwh = consumption_by_type.get('rural', 0)
    # Convert to GWh
    urban_gwh = urban_kwh / 1e6
    rural_gwh = rural_kwh / 1e6
    # Print the formatted results
    print(f"  - Urban: {urban_gwh:,.2f} GWh")
    print(f"  - Rural: {rural_gwh:,.2f} GWh")

    return grid_with_predictions

if __name__ == "__main__":
    sys.path.insert(1, '../../')
    import config
    grid = read_csv(config.RESIDENTIAL_GRID_FILE)
    grid = estimate_electricity_rwi_link(grid, config)
    # grid.to_csv(config.RESIDENTIAL_GRID_FILE,index=False)

