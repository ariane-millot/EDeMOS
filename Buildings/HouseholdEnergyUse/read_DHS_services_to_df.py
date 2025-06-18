"""
Script to read DHS household data and export it into a .csv


# This script reads data files from the DHS survey
# for instance https://dhsprogram.com/data/dataset/Kenya_Standard-DHS_2022.cfm
# and then outputs selected columns to a csv file
"""

from pandas import read_fwf
import numpy as np
import pandas as pd

# Check if we are running the notebook directly, if so move workspace to parent dir
import sys
import os
currentdir = os.path.abspath(os.getcwd())
if os.path.basename(currentdir) != 'EDeMOS':
  sys.path.insert(0, os.path.dirname(currentdir))
  os.chdir('..')
  print(f'Move to {os.getcwd()}')

import config


# Choose the file
filepath = config.DHS_FOLDER / f'{config.DHS_SERVICES_SURVEY_FILE }.DTA'
namefilepath = config.DHS_FOLDER / f'{config.DHS_SERVICES_SURVEY_FILE}.DO'

# When labels are repeated, use the method below https://stackoverflow.com/questions/31782283/loading-stata-file-categorial-values-must-be-unique
with pd.io.stata.StataReader(filepath) as sr:
    value_labels = sr.value_labels()

df = pd.read_stata(
    filepath,
    convert_categoricals=False,
)

for col in value_labels:
    if col.lower() in df.columns:
        df[col.lower()] = df[col.lower()].replace(value_labels[col])

# label variable v005     "Women's individual sample weight (6 decimals)"
# label variable v024     "Region"
# label variable v025     "Type of place of residence"
# label variable v704     "Husband/partner's occupation"
# label variable v704a    "Husband/partner worked in last 7 days/12 months"
# label variable v705     "Husband/partner's occupation (grouped)"
# label variable v714     "Respondent currently working"
# label variable v714a    "Respondent has a job, but currently absent"
# label variable v716     "Respondent's occupation"
# label variable v717     "Respondent's occupation (grouped)"
# label variable v719     "Respondent works for family, others, self"
# label variable v721     "NA - Respondent works at home or away"
labels_services = ['v005','v024', 'v025', 'v704', 'v704a', 'v705', 'v714', 'v714a', 'v716','v717', 'v719', 'v721']
df_services = df[labels_services]
# Read the names of the columns from the .DO file and convert to dictionary
do = read_fwf(namefilepath, skiprows=2, encoding='latin-1')
col_dict = dict(zip(do['caseid'], do['"Case Identification"']))
df_services = df_services.rename(columns=col_dict)

# Calculate the occupation share within each region and per type of area - women
occupation_share_women = df.groupby(['v024', 'v025', 'v717'])['v005'].sum().unstack(fill_value=0)/1000000
occupation_share_women = occupation_share_women.div(occupation_share_women.sum(axis=1), axis=0) * 100  # Calculate share as percentage
occupation_share_women = occupation_share_women.applymap('{:.2f}'.format)
print(occupation_share_women)

# Calculate the occupation share within each region - women - to compare with table 3.7.1 from the DHS report
occupation_share_women_DHSreport = df.groupby(['v024', 'v717'])['v005'].sum().unstack(fill_value=0)/1000000
occupation_share_women_DHSreport = occupation_share_women_DHSreport.drop(['not working'], axis=1)
occupation_share_women_DHSreport = occupation_share_women_DHSreport.div(occupation_share_women_DHSreport.sum(axis=1), axis=0) * 100  # Calculate share as percentage
occupation_share_women_DHSreport = occupation_share_women_DHSreport.applymap('{:.2f}'.format)
print(occupation_share_women_DHSreport)

# Calculate the share of each occupation within each region and per type of area = men
occupation_share_men = df.groupby(['v024', 'v025', 'v705'])['v005'].sum().unstack(fill_value=0)/1000000
occupation_share_men = occupation_share_men.div(occupation_share_men.sum(axis=1), axis=0) * 100  # Calculate share as percentage
occupation_share_men = occupation_share_men.applymap('{:.2f}'.format)
print(occupation_share_men)

# Save the two DataFrame to a CSV file
occupation_share_men.to_csv(config.DHS_FOLDER / config.DHS_EMPLOYEE_MEN_CSV)
occupation_share_women.to_csv(config.DHS_FOLDER / config.DHS_EMPLOYEE_WOMEN_CSV)