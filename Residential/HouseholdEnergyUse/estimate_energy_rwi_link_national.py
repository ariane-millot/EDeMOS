from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns

from estimate_energy_perhh_DHS import compute_energy_perhh_DHS

data_folder = '../Data/DHSSurvey/'
figures_folder = '../Figure/'
make_figure = True

elas = 0.35  # choose elasticity value for the country
compute_energy_perhh_DHS(elas=elas)  # Run the script to assess energy consumption of households in the DHS dataset

infile = data_folder + 'household_data.csv'  # Read file containing data from DHS survey of households
dataDHS = read_csv(infile)

wealth_index = 1e-5 * dataDHS["Wealth index factor score for urban/rural (5 decimals)"].to_numpy(float)
weight = 1e-6 * dataDHS['Household sample weight (6 decimals)'].to_numpy(float)
energy_use = dataDHS["Energy Use Elasticity"].to_numpy(float)  # Choose if assessed energy with elas is used or not
# energy_use = dataDHS["Energy Use"].to_numpy(float)
province = dataDHS['Province'].to_numpy(int)

region_type = ['urban', 'rural']

# Parameters for the graphs
min_wealth = np.floor(wealth_index.min())
max_wealth = np.ceil(wealth_index.max())
print(min_wealth, max_wealth)
# xl = np.array([-2, 2])  # Limits for x-axis of plots (wealth index)
xl = np.array([min_wealth, max_wealth])  # Limits for x-axis of plots (wealth index)
bx = np.arange(xl[0], xl[1], 0.1)  # Bins for histograms on x-axis
yl = np.array([0, 2999])  # Limits for y-axis on scatter plots (energy use)
al = 0.3  # Alpha value for points in figures
x_axis_cells_results_option = False
letters = ['(a)', '(b)']

hh_cells_results_available = True  # Change to False if this is the first run of the analysis
if hh_cells_results_available is True:
    infile = '../../Outputs/data.csv'  # Read file containing the mean wealth index ("rwi") of each hexagon on map
    data_hh = read_csv(infile)
    N = data_hh.shape[0]
    # print(N)
    energy_demand = np.zeros((2, N))  # Array to save energy demand estimates
    # print(energy_demand)


outputfile = '../Data/DHSSurvey/dataDHS_group.csv'
# Create an empty DataFrame
data_output = pd.DataFrame()

Nb = 100  # Number of points to used to map wealth index to energy use

for i in range(2):

    this_region_type = dataDHS["Type of place of residence"] == i + 1
    column_name = 'HH_' + region_type[i].lower()
    in_region_type = np.flatnonzero(this_region_type)

    rwi_DHS = wealth_index[in_region_type]
    eu = energy_use[in_region_type]
    w = weight[in_region_type]

    Nh = rwi_DHS.size
    o = np.argsort(rwi_DHS)  # Find indices that order the survey households by ascending rwi

    # Create Nb overlapping groups of points with ascending average rwi
    Ng = min(int(Nh / 4), 1000)  # define the size of each group of DHS data points
    print(Ng)
    step = int((Nh - Ng) / (Nb - 1))  # define the increment between each group
    group = np.zeros((Nb, Ng), dtype=int)
    for k in range(Nb):  # creates overlapping group as step < Ng
        Ni = step * k + Ng
        group[k, :] = o[step * k:Ni]
    group_high_rwi = o[Ni:]

    # Calculation average rwi, average energy use, and average weighting for each group
    rwi_group = np.append(np.sum(rwi_DHS[group] * w[group], axis=1) / np.sum(w[group], axis=1),
                          np.sum(rwi_DHS[group_high_rwi] * w[group_high_rwi]) / np.sum(w[group_high_rwi]))
    eu_group = np.append(np.sum(eu[group] * w[group], axis=1) / np.sum(w[group], axis=1),
                         np.sum(eu[group_high_rwi] * w[group_high_rwi]) / np.sum(w[group_high_rwi]))

    # rwi_group = np.append(np.mean(rwi_DHS[group],axis=1), np.mean(rwi_DHS[group_high_rwi]))
    # eu_group = np.append(np.mean(eu[group],axis=1),np.mean(eu[group_high_rwi]))

    w_group = np.append(np.mean(w[group], axis=1), np.mean(w[group_high_rwi]))

    # Write the results in dataDHS_group
    data_output['rwi_group_' + region_type[i]] = rwi_group
    data_output['eu_group_' + region_type[i]] = eu_group
    data_output['w_group_' + region_type[i]] = w_group

    if hh_cells_results_available is True:
        # Create filter to identify map regions (hexagons) of the relevant type and province
        include = (data_hh[column_name] > 0)

        # Allocate estimated average household energy demand to hexagon subregions
        # by interpolating between running average of survey data
        energy_demand[i, include] = np.interp(data_hh['rwi'][include], rwi_group, eu_group)

    if make_figure:
        palette = sns.color_palette()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5),
                                       gridspec_kw={'height_ratios': [1, 2]})

        # First subplot
        # Plots a weighted density histogram of rwi DHS values for survey households.
        ax1.hist(rwi_DHS, bins=bx, weights=w/w.sum()*100, density=False, edgecolor=palette[0],
                 facecolor='None', label='DHS households')
        # sns.kdeplot(x=rwi_DHS, weights=w/w.sum()*100, color=palette[0], ax=ax1, label='Survey households')
        ax1.set_xlim(xl)
        ax1.set_xticks([])
        # ax1.set_ylim(0, 80)
        ax1.set_ylabel('Household\ndensity (%)\n\n')

        if hh_cells_results_available is True:
            if x_axis_cells_results_option is True:
                # option to choose a different x-axis
                # Parameters for the graphs
                min_rwi = np.floor(data_hh['rwi'][include].min())
                max_rwi = np.ceil(data_hh['rwi'][include].max())
                print(min_rwi, max_rwi)
                xl = np.array([min_rwi, max_rwi])  # Limits for x-axis of plots (wealth index)
                ax1.set_xlim(xl)
            # Plots a density histogram of rwi values for our dataset
            ax1.hist(data_hh['rwi'][include],
                     bins=bx, weights=data_hh[column_name][include]/data_hh[column_name][include].sum()*100,
                     density=False, edgecolor=palette[4], facecolor='None', label='Dataset households')
            # sns.kdeplot(x=data_hh['rwi'][include], weights=data_hh[column_name][include]/d
            # ata_hh[column_name][include].sum()*100,
            #             color=palette[4], ax=ax1, label='Dataset households')
        ax1.legend()

        # Second subplot
        ax2.set_xlim(xl)
        # ax2.set_ylim(yl)
        scaling = 3
        # Plots a scatter plot of rwi vs. energy use for survey households from DHS data
        ax2.scatter(rwi_DHS, eu, s=w*5*scaling, alpha=al, c=[palette[0]], edgecolors='None', label='DHS households')
        ax2.text(xl[0], yl[-1], '\n  ', va='top')
        # # Add the average rwi vs energy use for the region
        # ax2.plot(rwi_DHS.mean(), eu.mean(), '+', color=palette[2], ms=20)
        ax2.set_ylabel('Household annual\nelectricity consumption (kWh)')
        ax2.set_xlabel('Wealth index')

        # Scatter plot of national values
        ax2.plot(rwi_group, eu_group, '-', color='grey', alpha=0.3)
        # Scatter plot of group values
        ax2.scatter(rwi_group, eu_group, s=w_group*10*scaling, marker='d', alpha=al*2, c='None', edgecolors=palette[1],
                    label='Artificial groups of DHS households')
        ax2.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if hh_cells_results_available is True:
            # Scatter plot of rwi values for our dataset vs energy demand
            ax2.scatter(data_hh['rwi'][include], energy_demand[i, include],
                        s=data_hh[column_name][include]/data_hh[column_name][include].sum()*100*10*scaling, alpha=al,
                        marker='o', c=[palette[4]],
                        edgecolors=palette[4], label='Dataset households')
        ax2.legend()

        # Put a legend below current axis
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels, title='Key to symbols', loc='lower center',
                   bbox_to_anchor=(0.5, -0.6),
                   fancybox=False, ncol=4, facecolor='white', framealpha=1)

        if hh_cells_results_available is True:
            outfile = f'household_groups_{region_type[i].lower()}_withweight_elas{elas}_withWorldpopData_newvalues.png'
        else:
            outfile = f'household_groups_{region_type[i].lower()}_withweight_elas{elas}_woWorldpopData_newvalues.png'
        pathlib.Path(figures_folder).mkdir(exist_ok=True)
        fig.suptitle(f'{letters[i]} {region_type[i].capitalize()}')
        plt.tight_layout()
        plt.savefig(figures_folder + outfile, dpi=300)
        print('Created ' + outfile)
        # plt.show()
        plt.close()

    # # Add the assessed energy use in the output file
    # column_out = 'Mean ' + region_type[i].lower() + ' household energy demand, kWh/year'
    # data_hh[column_out] = energy_demand[i, :]

# Save the data
data_output.to_csv(outputfile, index=False)
