from pandas import read_csv
import numpy as np


# Update the household_data file based on information in appliance_energy_use.csv
def compute_energy_perhh_DHS(elas=2, data_folder='../Data/DHSSurvey/'):

    # Read-in the data on appliances and energy tiers
    data_apps = read_csv(data_folder + './appliance_energy_use.csv', header=1)
    # Identify columns that give appliance energy consumption
    cols = [c for c in data_apps.columns if "consumption" in c]
    # Read columns into 2-d array for appliance consumption
    energy_cons = data_apps[cols].to_numpy(float)
    appliance = np.array(data_apps['Appliance'])  # Appliance names

    # Read-in the data from the survey of households
    infile = data_folder + 'household_data.csv'
    data = read_csv(infile)
    Nh = data.shape[0]
    print('Read data on', Nh, 'survey households')
    # Read columns into 2d array on appliance usage
    appliance_use = data[appliance].to_numpy(int)

    # Create array ready to store energy use estimates
    energy_use = np.zeros(Nh)
    energy_use_elas = np.zeros(Nh)

    # Create array to give the mapping between appliance usage and tier
    tier = np.array([0, 0, 0, 1, 2, 2, 3, 4])

    # Create filter to avoid including houses that don't even have electricity
    has_electricity = appliance_use[:, 0] > 0

    # set counter to follow tiers in loop below
    t = -1

    # define elasticity for en cons.
    rwi_col = 'Wealth index factor score for urban/rural (5 decimals)'
    rwi = 1e-5 * data[rwi_col].to_numpy(float)

    print('Estimating average energy use per household...')
    for i in range(tier.size):

        if tier[i] > t:  # Higher tier - update energy estimates for households in this tier or above

            # Decide which houses are in this tier based on their appliance usage
            in_tier = np.flatnonzero(has_electricity & (appliance_use[:, i] > 0))

            # Estimate energy use for each household by multiplying each appliance they use
            # by the energy consumption of that appliance relevant to the tier they're in
            # energy_use[in_tier] = np.sum(appliance_use[in_tier, :] * energy_cons[:, tier[i]], axis=1)
            energy_use[in_tier] = appliance_use[in_tier, :] @ energy_cons[:, tier[i]]

            # Estimate energy use for each household by multiplying each appliance they use
            # by the energy consumption of that appliance relevant to the tier they're in
            # and taking into account an elasticity factor
            # find the average rwi for all HH in the tier
            rwi_average_tier = rwi[in_tier].mean()
            # assess the energy consumption
            energy_use_elas[in_tier] = (
                    appliance_use[in_tier, :] @ energy_cons[:, tier[i]]
                    * (1 + elas * (rwi[in_tier]/rwi_average_tier))
            )
            # remove negative values
            energy_use_elas = np.clip(energy_use_elas, a_min=0, a_max=None)

            # The first pass through this section allocates all houses with tier 1 consumption
            # the second overwrites the houses with tier 2 appliances using tier 2 consumption levels
            # etc. up to tier 5

            # N.B. The averages printed out below are NOT the average for houses in that tier group
            # they are just for the purpose of tracking and sense-checking the code
            print('Tier', tier[i]+1, '('+','.join(appliance[tier == tier[i]])+') = {:,.1f} kWh/y'.format(np.mean(energy_use[in_tier])))

            t = tier[i]

    # Write or overwrite column in data file with estimated energy use values
    data['Energy Use'] = energy_use
    data['Energy Use Elasticity'] = energy_use_elas
    data.to_csv(infile, index=None)
    print('Written energy use estimates to', infile)


if __name__ == "__main__":
    compute_energy_perhh_DHS()