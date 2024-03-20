from pandas import read_csv, read_excel
import numpy as np
import matplotlib.pyplot as plt

data_folder = '../Data/DHSSurvey/'
figures_folder = '../Figures/'

infile = data_folder + 'household_data.csv' # Read file containing data from survey of households
dataDHS = read_csv(infile)

wealth_index = 1e-5 * dataDHS["Wealth index factor score for urban/rural (5 decimals)"].to_numpy(float)
weight = 1e-6 * dataDHS['Household sample weight (6 decimals)'].to_numpy(float)
energy_use = dataDHS["Energy Use"].to_numpy(float)
province = dataDHS['Province'].to_numpy(int)

region_type = ['Urban', 'Rural']

province_list = np.array(["National",
                 "Central",
                 "Copperbelt",
                 "Eastern",
                 "Luapula",
                 "Lusaka",
                 "Muchinga",
                 "Northern",
                 "North Western",
                 "Southern",
                 "Western"])

Nprov = province_list.size

xl = np.array([-2, 3.99])  # Limits for x-axis of plots (wealth index)
bx = np.arange(xl[0], xl[1], 0.4)  # Bins for histograms on x-axis
yl = np.array([0, 2999])  # Limits for y-axis on scatter plots (energy use)


infile = '../../data.csv'  # Read file containing the estimated mean wealth index ("rwi") of each hexagon on map
data_subregions = read_csv(infile)
N = data_subregions.shape[0]
# N = 1 # only national level

Nb = 20  # Number of points to used to map wealth index to energy use

energy_demand = np.zeros((2, N))  # Array to save energy demand estimates

make_figure = True
if make_figure:
    al = 0.6  # Alpha value for points in figures

for i in range(2):

    this_region_type = dataDHS["Type of place of residence"] == i + 1

    if make_figure:
        fig = plt.figure(figsize=(14, 8))
        plt.subplots_adjust(left=0.06, bottom=0.07, right=0.94, top=0.96, hspace=0, wspace=0.03)
        column_name = 'HH_' + region_type[i].lower()

        axk = plt.subplot(7, 1, 4)
        axk.set_axis_off()
        axk.set_xlim(0, 1)
        axk.set_ylim(0, 1)
        xk = 0.1
        dxk = 0.25
        yk = 0.3

    for j in range(Nprov):

        # Create filter to identify survey households of the relevant type and province
        if j > 0:
            in_province = np.flatnonzero(this_region_type & (province == j))
        else:
            in_province = np.flatnonzero(this_region_type)

        rwi_DHS = wealth_index[in_province]
        eu = energy_use[in_province]
        w = weight[in_province]

        Nh = rwi_DHS.size
        o = np.argsort(rwi_DHS)  # Find indices that order the survey households by ascending rwi
        # Create filter to identify map regions (hexagons) of the relevant type and province
        include = (data_subregions[column_name] > 0) & (data_subregions["NAME_1"] == province_list[j].replace(' ', '-'))

        # Create Nb overlapping groups of points with ascending average rwi
        Ng = min(int(Nh / 4), 1000)  # define the size of each group of DHS data points
        step = int((Nh - Ng) / (Nb - 1))  # define the increment between each group
        group = np.zeros((Nb, Ng), dtype=int)
        for k in range(Nb):  # creates overlapping group as step < Ng
            Ni = step * k + Ng
            group[k, :] = o[step * k:Ni]
        group_high_rwi = o[Ni:]

        # Calculation average rwi, average energy use, and average weighting for each group
        rwi_group = np.append(np.mean(rwi_DHS[group], axis=1), np.mean(rwi_DHS[group_high_rwi]))
        eu_group = np.append(np.mean(eu[group], axis=1), np.mean(eu[group_high_rwi]))
        w_group = np.append(np.mean(w[group], axis=1), np.mean(w[group_high_rwi]))

        # Allocate estimated average household energy demand to hexagon subregions
        # by interpolating between running average of survey data
        energy_demand[i, include] = np.interp(data_subregions['rwi'][include], rwi_group, eu_group)

        if make_figure:

            if j > 0:
                row = int((j - 1) / 5)
                col = j - 1 - 5 * row

                # First subplot
                ax1 = plt.subplot2grid((7, 5), (row * 4, col), 1, 1)
                # Plots a weighted density histogram of rwi DHS values for survey households.
                plt.hist(rwi_DHS, bins=bx, weights=w, density=True, alpha=al, edgecolor='b', facecolor='None')
                plt.xlim(xl)
                plt.xticks([])
                plt.ylim(0, 1.7)
                # Plots a density histogram of rwi values for our dataset with orange shading.
                ax1.hist(data_subregions['rwi'][include],
                         bins=bx, weights=data_subregions[column_name][include],
                         density=True, edgecolor='None', facecolor='orange')

                # Second subplot
                ax2 = plt.subplot2grid((7, 5), (1 + row * 4, col), 2, 1)
                plt.xlim(xl)
                plt.ylim(yl)
                # Plots a scatter plot of rwi vs. energy use for survey households from DHS data with light green dot
                plt.scatter(rwi_DHS, eu, s=w, c='c', alpha=al, edgecolors='None')
                plt.text(xl[0], yl[-1], '\n  ' + province_list[j], va='top')
                # Add the average rwi vs energy use for the region
                plt.plot(rwi_DHS.mean(), eu.mean(), '+', color='g', ms=20)

                if col in [0, 4]:
                    plt.ylabel('Household annual energy use, kWh')
                    if col == 4:
                        ax1.tick_params(axis='y', right=True, labelright=True, labelleft=False)
                        ax2.tick_params(axis='y', right=True, labelright=True, labelleft=False)
                        ax2.yaxis.set_label_position("right")
                else:
                    ax1.set_yticklabels([])
                    ax2.set_yticklabels([])
                plt.xlabel('Wealth index')

                # Scatter plot of national values
                plt.plot(wi_nat, eu_nat, '-', color='grey', alpha=0.3)
                # Scatter plot of group values
                plt.scatter(rwi_group, eu_group, s=w_group, alpha=al, marker='d', c='None', edgecolors='b')
                # Scatter plot of rwi values for our dataset vs energy demand for subregions with orange dots
                plt.scatter(data_subregions['rwi'][include], energy_demand[i, include], alpha=al, marker='o', s=1,
                            c='orange', edgecolors='None')

            else:
                wi_nat = rwi_group.copy()
                eu_nat = eu_group.copy()

    if make_figure:
        # Make key to symbols
        axk.text(0, yk, 'Key to symbols:', va='center')
        axk.plot(xk, yk, 'o', mfc='c', mec='None', alpha=al)
        axk.text(xk, yk, '  Survey households', va='center')
        xk += dxk
        axk.plot(xk, yk, 'd', mfc='None', mec='b', alpha=al)
        axk.text(xk, yk, '  Artifical groups of survey households', va='center')
        xk += dxk
        axk.plot(xk, yk, 's', mec='None', mfc='orange', alpha=al)
        axk.text(xk, yk, '  Real groups of households from map', va='center')

        outfile = 'household_groups_' + region_type[i].lower() + '_regions.png'
        plt.savefig(figures_folder + outfile)
        print('Created ' + outfile)
        plt.close()

    # Add the assessed energy use in the output file
    column_out = 'Mean ' + region_type[i].lower() + ' household energy demand, kWh/year'
    data_subregions[column_out] = energy_demand[i, :]
# Write estimated energy use into the file containing the map region (hexagon) data
data_subregions.to_csv(infile, index=False)
print('Written energy estimates to ', infile)

