"""
Script to read DHS household data and export it into a .csv


# This script reads data files from the DHS survey
# for instance https://dhsprogram.com/data/dataset/Kenya_Standard-DHS_2022.cfm
# and then outputs selected columns to a csv file
"""

from pandas import read_fwf
import pandas as pd

# Check if we are running the notebook directly, if so move workspace to parent dir
import sys
import os
startdir = os.getcwd()
currentdir = os.path.abspath(startdir)
while os.path.basename(currentdir) != 'EDeMOS':
  sys.path.insert(0, os.path.dirname(currentdir))
  os.chdir('..')
  currentdir = os.path.abspath(os.getcwd())
print(f'Moved to {currentdir}')

import config

# Choose the file
filepath = config.DHS_FOLDER / f'{config.DHS_HH_SURVEY_FILE}.DTA'
namefilepath = config.DHS_FOLDER / f'{config.DHS_HH_SURVEY_FILE}.DO'

# When labels are repeated, use the method below https://stackoverflow.com/questions/31782283/loading-stata-file-categorial-values-must-be-unique
with pd.io.stata.StataReader(filepath) as sr:
    value_labels = sr.value_labels()

df = pd.read_stata(
    filepath,
    convert_categoricals=False,
)

# Use if you want to read the text version of the data
# print(value_labels)
for col in value_labels:
    if col.lower() in df.columns:
        df[col.lower()] = df[col.lower()].replace(value_labels[col])

# select labels from the .do file (see config)
df = df[config.labels_hh]
# Read the names of the columns from the .DO file and convert to dictionary
do = read_fwf(namefilepath, skiprows=2, encoding='latin-1')
col_dict = dict(zip(do['hhid'], do['"Case Identification"']))
print(list(col_dict.items())[:5] )# Show the first 5 items)

# Rename the columns
df_hh = df.rename(columns=col_dict)

# Strip double quotes and any leading or trailing whitespace from the column names
df_hh.columns = df_hh.columns.str.replace('"', '').str.strip()

# Define the columns we want to change and what their new names will be.
old_to_new_names = config.DHS_SURVEY_HH_old_to_new_names

# Get a list of the columns we need to modify (the ones with 'yes'/'no')
columns_to_change = list(old_to_new_names.keys())

# Check which of these columns actually exist in the DataFrame
columns_to_change_existing = [col for col in columns_to_change if col in df_hh.columns]

# Replace yes no by numerical values
replacement_map = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0}
df_hh[columns_to_change_existing] = df_hh[columns_to_change_existing].replace(replacement_map)

# Rename the columns to their shorter names
df_hh = df_hh.rename(columns=old_to_new_names)

print(df_hh.head())
# Write the data for useful columns only in .csv format
df_hh.to_csv(config.DHS_FOLDER / config.DHS_HOUSEHOLD_DATA_CSV)

# Check if we moved workspace to parent dir, if so revert to original directory
if currentdir != os.path.abspath(startdir):
    os.chdir(startdir)
    print(f'Moved back to {startdir}')