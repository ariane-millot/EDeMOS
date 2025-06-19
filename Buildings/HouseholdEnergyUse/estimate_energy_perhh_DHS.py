from pandas import read_csv
import numpy as np

# Check if we are running the notebook directly, if so move workspace to parent dir
import sys
import os
currentdir = os.path.abspath(os.getcwd())
if os.path.basename(currentdir) != 'EDeMOS':
  sys.path.insert(0, os.path.dirname(currentdir))
  os.chdir('..')
  print(f'Move to {os.getcwd()}')

import config

dhs_data_folder = config.DHS_FOLDER
data_folder = config.RESIDENTIAL_DATA_PATH

# Update the household_data file based on information in appliance_energy_use.csv
def compute_energy_perhh_DHS(elas=0.4, nominal_household_size=4):

    # Read-in the data on appliances and energy tiers
    data_apps = read_csv(data_folder / 'appliance_energy_use.csv', header=1)
    # Identify columns that give appliance energy consumption
    cols = [c for c in data_apps.columns if "consumption" in c]
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
    tier = np.array([0, 0, 0, 1, 2, 2, 3, 4])

    # Create filter to avoid including houses that don't even have electricity
    has_electricity = appliance_use[:, 0] > 0

    # set counter to follow tiers in loop below
    t = -1

    print('Estimating average energy use per household...')
    for i in range(tier.size):

        if tier[i] > t:  # Higher tier - update energy estimates for households in this tier or above

            # Decide which houses are in this tier based on their appliance usage
            in_tier = np.flatnonzero(has_electricity & (appliance_use[:, i] > 0))

            # Estimate energy use for each household by multiplying each appliance they use
            # by the energy consumption of that appliance relevant to the tier they're in
            # energy_use[in_tier] = np.sum(appliance_use[in_tier, :] * energy_cons[:, tier[i]], axis=1)
            energy_use[in_tier] = appliance_use[in_tier, :] @ energy_cons[:, tier[i]]

            # The first pass through this section allocates all houses with tier 1 consumption
            # the second overwrites the houses with tier 2 appliances using tier 2 consumption levels
            # etc. up to tier 5

            # N.B. The averages printed out below are NOT the average for houses in that tier group
            # they are just for the purpose of tracking and sense-checking the code
            print('Tier', tier[i]+1, '('+','.join(appliance[tier == tier[i]])+') = {:,.1f} kWh/y'.format(np.mean(energy_use[in_tier])))

            t = tier[i]

    energy_use = energy_use*(household_size/nominal_household_size)**elas

    # Write or overwrite column in data file with estimated energy use values
    data['Electricity Use'] = energy_use
    data.to_csv(config.DHS_HOUSEHOLD_DATA_CSV, index=None)
    print('Written energy use estimates to', config.DHS_HOUSEHOLD_DATA_CSV)


if __name__ == "__main__":
    compute_energy_perhh_DHS()