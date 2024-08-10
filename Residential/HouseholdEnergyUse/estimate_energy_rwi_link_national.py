from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from estimate_energy_perhh_DHS import compute_energy_perhh_DHS

data_folder = '../Data/DHSSurvey/'
figures_folder = '../Figure/'

elas = 1  # choose elasticity value for the country
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
xl = np.array([-2, 3.99])  # Limits for x-axis of plots (wealth index)
bx = np.arange(xl[0], xl[1], 0.4)  # Bins for histograms on x-axis
yl = np.array([0, 2999])  # Limits for y-axis on scatter plots (energy use)

hh_cells_results_available = False  # Change to False if this is the first run of the analysis
if hh_cells_results_available is True:
    infile = '../../data.csv'  # Read file containing the estimated mean wealth index ("rwi") of each hexagon on map
    data_hh = read_csv(infile)
    N = data_hh.shape[0]
    # print(N)
    energy_demand = np.zeros((2, N))  # Array to save energy demand estimates
    # print(energy_demand)

Nb = 20  # Number of points to used to map wealth index to energy use


make_figure = True
if make_figure:
    al = 0.6  # Alpha value for points in figures

outputfile = '../Data/DHSSurvey/dataDHS_group.csv'
# Create an empty DataFrame
data_output = pd.DataFrame()


Nb = 100

for i in range(2):

    this_region_type = dataDHS["Type of place of residence"] == i + 1
    column_name = 'HH_' + region_type[i].lower()
    in_region_type = np.flatnonzero(this_region_type)

    rwi_DHS = wealth_index[in_region_type]
    eu = energy_use[in_region_type]
    w = weight[in_region_type]

    Nh = rwi_DHS.size
    o = np.argsort(rwi_DHS)  # Find indices that order the survey households by ascending rwi
    if hh_cells_results_available is True:
        # Create filter to identify map regions (hexagons) of the relevant type and province
        include = (data_hh[column_name] > 0)

    # Create Nb overlapping groups of points with ascending average rwi
    Ng = min(int(Nh / 4), 1000)  # define the size of each group of DHS data points
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
        # Allocate estimated average household energy demand to hexagon subregions
        # by interpolating between running average of survey data
        energy_demand[i, include] = np.interp(data_hh['rwi'][include], rwi_group, eu_group)

    if make_figure:
        fig = plt.figure(figsize=(14, 8))
        plt.subplots_adjust(left=0.06, bottom=0.07, right=0.94, top=0.96, hspace=0, wspace=0.03)

        axk = plt.subplot(4, 1, 4)
        axk.set_axis_off()
        axk.set_xlim(0, 1)
        axk.set_ylim(0, 1)
        xk = 0.1
        dxk = 0.25
        yk = 0.3

    if make_figure:
        # First subplot
        ax1 = plt.subplot2grid((4, 1), (0, 0), 1, 1)

        # Plots a weighted density histogram of rwi DHS values for survey households.
        plt.hist(rwi_DHS, bins=bx, weights=w, density=True, alpha=al, edgecolor='b', facecolor='None')
        plt.xlim(xl)
        plt.xticks([])
        plt.ylim(0, 1.7)

        if hh_cells_results_available is True:
            # Plots a density histogram of rwi values for our dataset with orange shading.
            ax1.hist(data_hh['rwi'][include],
                     bins=bx, weights=data_hh[column_name][include],
                     density=True, edgecolor='None', facecolor='orange')

        # Second subplot
        ax2 = plt.subplot2grid((4, 1), (1, 0), 2, 1)
        plt.xlim(xl)
        # plt.ylim(yl)
        # Plots a scatter plot of rwi vs. energy use for survey households from DHS data with light green dot
        plt.scatter(rwi_DHS, eu, s=w, c='c', alpha=al, edgecolors='None')
        plt.text(xl[0], yl[-1], '\n  ', va='top')
        # Add the average rwi vs energy use for the region
        plt.plot(rwi_DHS.mean(), eu.mean(), '+', color='g', ms=20)
        plt.ylabel('Household annual energy consumption, kWh')
        plt.xlabel('Wealth index')

        # Scatter plot of national values
        plt.plot(rwi_group, eu_group, '-', color='grey', alpha=0.3)
        # Scatter plot of group values
        plt.scatter(rwi_group, eu_group, s=w_group, alpha=al, marker='d', c='None', edgecolors='b')
        if hh_cells_results_available is True:
            # Scatter plot of rwi values for our dataset vs energy demand for subregions with orange dots
            plt.scatter(data_hh['rwi'][include], energy_demand[i, include], alpha=al, marker='o', s=1, c='orange',
                        edgecolors='None')

    if make_figure:
        # Make key to symbols
        axk.text(0, yk, 'Key to symbols:', va='center')
        axk.plot(xk, yk, 'o', mfc='c', mec='None', alpha=al)
        axk.text(xk, yk, '  Survey households', va='center')
        xk += dxk
        axk.plot(xk, yk, 'd', mfc='None', mec='b', alpha=al)
        axk.text(xk, yk, '  Artifical groups of survey households', va='center')
        xk += dxk
        if hh_cells_results_available is True:
            axk.plot(xk, yk, 's', mec='None', mfc='orange', alpha=al)
            axk.text(xk, yk, '  Real groups of households from map', va='center')

        outfile = f'household_groups_{region_type[i].lower()}_withweight_elas{elas}_woWorldpopData.png'
        pathlib.Path(figures_folder).mkdir(exist_ok=True)
        plt.savefig(figures_folder + outfile)
        print('Created ' + outfile)
        plt.show()
        plt.close()

    # # Add the assessed energy use in the output file
    # column_out = 'Mean ' + region_type[i].lower() + ' household energy demand, kWh/year'
    # data_hh[column_out] = energy_demand[i, :]

# Save the data
data_output.to_csv(outputfile, index=False)
