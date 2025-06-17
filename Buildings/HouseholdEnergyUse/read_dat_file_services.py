from pandas import read_fwf
import numpy as np
import pandas as pd

# This script reads data files from the DHS survey
# https://dhsprogram.com/data/dataset/Zambia_Standard-DHS_2018.cfm
# and then outputs selected columns to a csv file

# Folder where data files are located
data_folder = '../Data/DHSSurvey/ZMHR71FL/ZMHR71DT'

# Choose the file prefix.
filename = 'ZMHR71FL'

df = pd.read_stata(data_folder + "/" + filename + '.DTA')

# Read the names of the columns from the .DO file and convert to dictionary
do = read_fwf(data_folder + "\\" + filename+'.DO', skiprows=2, header=None, sep = "")
col_dict = dict(zip(do[2], do[3]))

df = df.rename(columns=col_dict)
df.head(2)
df.columns.shape[0]

# List of useful columns for this project
cols_required = ["Sample strata for sampling errors",
                 "Women's individual sample weight (6 decimals)",
                 "Stratification used in sample design",
                 "Region",
                 # "Number of household members",
                 "Type of place of residence",
                 "Respondent's occupation",
                 "Respondent's occupation (grouped)",
                 "Respondent works for family, others, self",
                 "NA - Respondent works at home or away",
                 "Husband/partner's occupation",
                 "Husband/partner's occupation (grouped)",
                 "Wealth index combined",
                 "Wealth index factor score combined (5 decimals)",
                 "Wealth index for urban/rural",
                 "Wealth index factor score for urban/rural (5 decimals)"]

# # Define a dictionary to map old names to new names
# old_to_new_names = {
#     "Has electricity": "Electricity",
#     "Has radio": "Radio",
#     "Has mobile telephone": "Mobile telephone",
#     "Has television": "Television",
#     "Has refrigerator": "Refrigerator",
#     "Has a computer": "Computer"
# }

# Write the data for useful columns only in .csv format
outfile = data_folder+'household_data.csv'
# data = data[cols_required].rename(columns=old_to_new_names)
data = data[cols_required]

# Write the data for useful columns only in .csv format
outfile = data_folder+'household_data_services.csv'
# data[cols_required].to_csv(outfile,index=False)
data.to_csv(outfile, index=False)
print('Written to ', outfile)
# N.B. The data will still need to be translated from numeric values to their meaning
# using the information in the .DO file

