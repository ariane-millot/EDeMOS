from pandas import read_csv
import numpy as np
import sys

# Update the household_data file based on information in appliance_energy_use.csv
def compute_energy_perhh_DHS(elas=0.4,nominal_household_size=4):

    # Read-in the data on appliances and energy tiers
    data_apps = read_csv(data_folder / config.APPLIANCE_ELECTRICITY_CONS, header=1)
    # Identify columns that give appliance energy consumption
    cols = [c for c in data_apps.columns if "Tier" in c]
    # Read columns into 2-d array for appliance consumption
    energy_cons = data_apps[cols].to_numpy(float)
    appliance = np.array(data_apps['Appliance'])  # Appliance names

    # Read-in the data from the survey of households
    data = read_csv(config.DHS_HOUSEHOLD_DATA_CSV)
    Nh = data.shape[0]
    print('Read data on', Nh, 'survey households')
    # Read columns into 2d array on appliance usage
    appliance_use = data[appliance].to_numpy(int)
    household_size = data['Number of household members']

    # Create array ready to store energy use estimates
    energy_use = np.zeros(Nh)

    # Create array to give the mapping between appliance usage and tier
    tier = np.array([0, 0, 0, 1, 2, 2, 2, 3, 4])

    # Create filter to avoid including houses that don't even have electricity
    has_electricity = appliance_use[:, 0] > 0

    print('Estimating average energy use per household...')

    # Step 1: Determine the maximum tier for each household in a single, vectorized operation.
    # We create a matrix where each cell is the tier of an appliance if the household owns it, or 0 otherwise.
    # Then, we find the maximum tier across all appliances for each household (row).
    household_appliance_tiers = appliance_use * tier
    max_tier_per_hh = np.max(household_appliance_tiers, axis=1)

    # Step 2: Calculate energy use based on the determined tier for each household.
    # We loop through the unique tier values (0, 1, 2, 3, 4) instead of all appliances.
    unique_tiers = np.unique(max_tier_per_hh)

    for t_val in unique_tiers:
        # Find all households whose max tier is the current tier `t_val`
        # and who have electricity.
        in_this_tier = (max_tier_per_hh == t_val) & has_electricity

        # Get the appliance ownership for only these households
        appliances_in_tier = appliance_use[in_this_tier]

        # Calculate their energy using the consumption values for this specific tier
        # and assign it to the correct slice of the energy_use array.
        energy_use[in_this_tier] = appliances_in_tier @ energy_cons[:, t_val]

        # Get appliance names for this tier for the print statement
        tier_appliance_names = appliance[tier == t_val]
        print(f'Tier {t_val+1} ({",".join(tier_appliance_names)}) = {np.mean(energy_use[in_this_tier]):,.1f} kWh/y')


    # --- Configuration for Tier Override Rules ---
    # Define rules here. The code will process them from highest tier to lowest.
    # A household will only be assigned the first (highest-tier) rule it matches.
    override_rules = [
        {'column': 'Source of drinking water', 'value': 'yes', 'tier': 3},
        {'column': 'Type of toilet facility', 'value': 'flush', 'tier': 3},
        {'column': 'Has motorcycle/scooter', 'value': 'yes', 'tier': 4},
        {'column': 'Has car/truck', 'value': 'yes', 'tier': 5},
        {'column': 'Type of cooking fuel', 'value': 'electricity', 'tier': 5},

    ]

    print('\nApplying flexible override rules to adjust tiers...')

    # Sort rules by tier in descending order to ensure priority
    sorted_rules = sorted(override_rules, key=lambda r: r['tier'], reverse=True)

    # Keep track of households that have already been assigned a tier by an override rule
    overridden_households = np.zeros(Nh, dtype=bool)

    # Check if rule columns exist in the dataframe
    for rule in sorted_rules:
        if rule['column'] not in data.columns:
            print(f"Warning: Column '{rule['column']}' from override rules not found in data. Skipping rule.")
            continue # Skip to the next rule

        # The tier number (e.g., 5) corresponds to the consumption column index (e.g., 4)
        tier_index = rule['tier'] - 1
        if not (0 <= tier_index < energy_cons.shape[1]):
            print(f"Warning: Tier {rule['tier']} is invalid. Skipping rule.")
            continue

        # Identify households that match the current rule AND have not been overridden yet
        rule_matches = (data[rule['column']] == rule['value']).values
        households_to_update = rule_matches & ~overridden_households

        if np.any(households_to_update):
            # For the selected households, get their appliance ownership
            update_appliance_use = appliance_use[households_to_update, :]

            # Recalculate their energy use using the fixed tier's consumption values
            new_energy = update_appliance_use @ energy_cons[:, tier_index]

            # Overwrite their energy use in the main array
            energy_use[households_to_update] = new_energy

            # Mark these households as overridden so they won't be changed by a lower-tier rule
            overridden_households |= households_to_update

            print(f"Updated {np.sum(households_to_update)} households to Tier {rule['tier']} based on rule: '{rule['column']}' is '{rule['value']}'")


    # Scale energy use by household size
    energy_use = energy_use*(household_size/nominal_household_size)**elas

    # Write or overwrite column in data file with estimated energy use values
    data[config.DHS_ELEC_KWH_ASSESSED_SURVEY] = energy_use
    data.to_csv(config.DHS_HOUSEHOLD_DATA_CSV, index=None)
    print('Written energy use estimates to', config.DHS_HOUSEHOLD_DATA_CSV)


# Check if we are running the notebook directly, if so import config
if __name__ == "__main__":

    sys.path.insert(1, '../../')

    import config

    dhs_data_folder = config.DHS_FOLDER
    data_folder = config.RESIDENTIAL_DATA_PATH

    compute_energy_perhh_DHS()
