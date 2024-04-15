from pandas import read_fwf
import numpy as np

# This script reads data files from the DHS survey
# https://dhsprogram.com/data/dataset/Zambia_Standard-DHS_2018.cfm
# and then outputs selected columns to a csv file

# Folder where data files are located
data_folder = '../Data/DHSSurvey/ZMIR71FL/'

# Choose the file prefix.
filename = 'ZMIR71FL'

# N is the number of columns in the file.
# This currently needs to be found before running this script by inspecting the files in a text editor
N = 4000

# Read the .DCT file which specifies which characters of the .DAT file encode which columns
# replace - with 3 spaces in the .dct file before running the code
dct = read_fwf(data_folder+filename+'.DCT', skiprows=2, header=None, nrows=N)
print('Read .dct file: shape =', dct.shape)
# Read the names of the columns from the .DO file
do = read_fwf(data_folder+filename+'.DO', skiprows=2, header=None, nrows=N)
print('Read .do file: shape =', do.shape)

# Append the column names to the character positions
dct = dct.merge(do, left_on=1, right_on=2, how='left')

# Make array of column names
cols = np.array(dct[dct.columns[[4, 5]]])

# Reduce the character positions by 1 because python starts counting from 0 (not from 1).
cols[:, 0]-=1

# Now read .DAT file containing encoded data
data = read_fwf(data_folder+filename+'.DAT',header=None, colspecs=list(map(tuple,cols)))

# Get rid of any duplicated quotation marks in column names
data.columns = np.char.strip(dct.iloc[:,-1].to_numpy(str),'"')

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

