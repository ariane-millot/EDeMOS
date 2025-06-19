from pyproj import CRS
from pathlib import Path

ROOT_DIR = Path( __file__ ).parent.absolute()

# -----------------------------------------------------------------------------
# AREA OF INTEREST CHOICE -- to update with the country
# -----------------------------------------------------------------------------
COUNTRY = "Zambia"
# Define area of interest
AREA_OF_INTEREST = "COUNTRY"  # Can be "COUNTRY" or a specific region like "Copperbelt"
ADMIN_GPKG = "gadm41_ZMB.gpkg"
YEAR = 2019 # year of analysis

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FOLDER = ROOT_DIR / "Data"
# Input paths
ADMIN_PATH = DATA_FOLDER / "admin"

# Output paths
OUTPUT_DIR = ROOT_DIR / "Outputs"
# Ensure all folders for output files exist
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# COORDINATE REFERENCE SYSTEMS
# -----------------------------------------------------------------------------
CRS_WGS84 = CRS("EPSG:4326") # Original WGS84 coordinate system
CRS_PROJ = CRS("EPSG:32736") # Projection system for the selected country - see http://epsg.io/ for more info
TARGET_CRS_METERS = "EPSG:32735" # Used for grid line buffering (UTM Zone 35S)

# -----------------------------------------------------------------------------
# PARAMETERS - HEXAGONS
# -----------------------------------------------------------------------------

# Admin boundaries
ADMIN_LAYER_REGION = "ADM_ADM_1"
ADMIN_LAYER_COUNTRY = "ADM_ADM_0"
ADMIN_REGION_COLUMN_NAME = "NAME_1"

# hexagon size
HEX_SIZE = 6 ## resolution info here https://h3geo.org/docs/core-library/restable

# HEXAGON FILE NAME
H3_GRID_HEX_SHP = "h3_grid_at_hex.shp" # Located in current OUTPUT_DIR

# BUFFER DISTANCE in meters.
buffer_distance_meters = 1000 # This should be larger than half the diagonal of a hexagon


# -----------------------------------------------------------------------------
# RESIDENTIAL PARAMETERS
# -----------------------------------------------------------------------------

# File paths

# Input paths
GRID_PATH = DATA_FOLDER / "Grid" # For MV/HV lines
ENERGY_BALANCE_PATH = DATA_FOLDER / "EnergyBalance"
RESIDENTIAL_DATA_PATH = ROOT_DIR / "Buildings" / "Data"
WORLDPOP_PATH = RESIDENTIAL_DATA_PATH / "WorldPop"
LIGHTING_PATH = RESIDENTIAL_DATA_PATH / "Lighting"
RWI_PATH = RESIDENTIAL_DATA_PATH / "WealthIndex"
FALCHETTA_PATH = RESIDENTIAL_DATA_PATH / "Falchetta_ElecAccess"
# GDP_PATH = RESIDENTIAL_DATA_PATH / "GDP")

# Output paths
RESIDENTIAL_OUTPUT_DIR = ROOT_DIR / "Buildings" / "Outputs" # As used in building_demand.py for dataHH_region.csv
FIGURES_DHS_FOLDER = ROOT_DIR / "Buildings" / "Figures" # As used in estimate_energy_rwi_link_national_new.py

# WorldPop files
# Link: https://apps.worldpop.org/peanutButter/
WP_BUILDINGS_COUNT_TIF = "ZMB_buildings_v2_0_count.tif"
WP_BUILDINGS_URBAN_TIF = "ZMB_buildings_v2_0_urban.tif"

# Lighting file
# # set_lightscore_sy_xxxx.tif: Predicted likelihood that a settlement is electrified (0 to 1)
# Link: http://www-personal.umich.edu/~brianmin/HREA/data.html
HREA_LIGHTING_TIF = "Zambia_set_lightscore_sy_2019.tif"

# RWI file
# Link: https://gee-community-catalog.org/projects/rwi/
# run the notebook Rasterize RWI.ipynb to generate rwi_map.tif
RWI_MAP_TIF = "rwi_map.tif"
RWI_FILE_CSV = "zmb_relative_wealth_index.csv"

# Falchetta Tiers file
# - Use QGIS to generate the raster file (*.tif)
# - EPSG:4326
FALCHETTA_TIERS_TIF = "Zambia_tiersofaccess_2018.tif"

# GDP file Kummu dataset
# Link https://www.nature.com/articles/sdata20184#Sec9
# GDP_PPP_TIF = "GDP_PPP_30arcsec_v3_band3_Zambia.tif"

# UN Energy Balance file
# https://data.un.org/SdmxBrowser/start
UN_ENERGY_BALANCE_CSV = "UNSD+DF_UNData_EnergyBalance+1.0_Zambia.csv"

# Grid line files
# Data available at https://datacatalog.worldbank.org/search/dataset/0040190/Zambia---Electricity-Transmission-Network
# or https://energydata.info/dataset/zambia-electrical-lines
MV_LINES_SHP = GRID_PATH / "Zambia - MVLines" / "Zambia - MVLines.shp"
HV_LINES_SHP = GRID_PATH / "Zambia - HVLines" / "HVLines.shp"

# Census data files
# The file should contain the following data, region, HH urban, rural, total, size of HH urban/rural
PROVINCE_DATA_AVAILABLE = True
CENSUS_ZAMBIA_PROVINCE_CSV = RESIDENTIAL_DATA_PATH / "Census" / "Census_Zambia.csv"
CENSUS_ZAMBIA_NATIONAL_CSV = RESIDENTIAL_DATA_PATH / "Census" /  "Census_Zambia_National.csv"

# DHS Survey related files (used by estimate_energy_rwi_link_national_new.py)
# 1. Run read_DHS_hh_to_df.py in HouseholdEnergyUse folder to generate the household_data.csv
# 2. Run estimate_energy_perhh_DHS.py in HouseholdEnergyUse folder
# For services:
# 1 Download the Individual Recode  XXXXDT.ZIP to generate employee_survey_(men/women).csv
# 2 Run: read_DHS_services_to_df.py in HouseholdEnergyUse folder
# 3 Download or add the 'pop15-49_share_{COUNTRY}.csv' file in the DHS folder
DHS_FOLDER = RESIDENTIAL_DATA_PATH / "DHS"
DHS_HOUSEHOLD_DATA_CSV = DHS_FOLDER / f'household_data_{COUNTRY}.csv'
DHS_EMPLOYEE_WOMEN_CSV = DHS_FOLDER / f'employee_survey_women_{COUNTRY}.csv'
DHS_EMPLOYEE_MEN_CSV = DHS_FOLDER / f'employee_survey_men_{COUNTRY}.csv'
DHS_WORKING_POP_SHARE_CSV = DHS_FOLDER / f'pop15-49_share_{COUNTRY}.csv'

# Ensure all folders for output files exist
RESIDENTIAL_OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DHS_FOLDER.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# DHS FILES PARAMETERS
# -----------------------------------------------------------------------------
DHS_HH_SURVEY_FILE = 'ZMHR71DT/ZMHR71FL'
DHS_SERVICES_SURVEY_FILE = 'ZMIR71DT/ZMIR71FL'

# Households labels
# Choose the labels to be selected from the .do file
#label variable hv005       "Household sample weight (6 decimals)"
#label variable hv009       "Number of household members",
#label variable hv022       "Sample strata for sampling errors"
#label variable hv023       "Stratification used in sample design"
#label variable hv024       "Province"
#label variable hv025       "Type of place of residence"
#label variable hv206       "Has electricity",
#label variable hv207       "Has radio"
#label variable hv208       "Has television"
#label variable hv209       "Has refrigerator"
#label variable hv243a      "Has mobile telephone"
#label variable hv243e      "Has a computer"
# label variable sh121f   "Access to Internet"
# label variable sh121j   "Washing machine"
# label variable sh121k   "Air conditioner"
# label variable sh121l   "Generator"
# label variable sh121m   "Microwave"
#label variable hv270       "Wealth index combined"
#label variable hv270a      "Wealth index for urban/rural"
#label variable hv271       "Wealth index factor score combined (5 decimals)"
#label variable hv271a      "Wealth index factor score for urban/rural (5 decimals)"
labels_hh = ['hv005','hv009', 'hv022', 'hv023', 'hv024', 'hv025', 'hv206', 'hv207', 'hv208', 'hv209', 'hv243a',
             'hv243e', 'sh121f', 'sh121j', 'sh121k', 'sh121l', 'sh121m',
             'hv270', 'hv270a', 'hv271', 'hv271a']

# Define the columns we want to change in the DHS data and what their new names will be.
DHS_SURVEY_HH_old_to_new_names = {
    "Has electricity": "Electricity",
    "Has radio": "Radio",
    "Has mobile telephone": "Mobile telephone",
    "Has television": "Television",
    "Has refrigerator": "Refrigerator",
    "Has a computer": "Computer",
    "Access to Internet": "Internet",
    "Washing machine": "Washing machine",
    "Air conditioner": "Air conditioner",
    "Generator": "Generator",
    "Microwave": "Microwave",
}

# -----------------------------------------------------------------------------
# PARAMETERS ENERGY BALANCE
# -----------------------------------------------------------------------------

# UN Energy Balance codes and year
UN_ELEC_CODE = "B07_EL"
UN_HH_TRANSACTION_CODE = "B50_1231"
UN_SERVICES_TRANSACTION_CODE = "B49_1235"
UN_OTHER_TRANSACTION_CODE = "B51_1234" # Other consumption not elsewhere specified
UN_ENERGY_YEAR = YEAR


# -----------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# -----------------------------------------------------------------------------

# Residential demand parameters
THRESHOLD_ELEC_ACCESS_URBAN = 0.9
THRESHOLD_ELEC_ACCESS_RURAL = 0.1
MV_LINES_BUFFER_DIST = 500 # meters
HV_LINES_BUFFER_DIST = 500 # meters
NB_OF_HH_PER_RES_BUILDING_URBAN = 1.1 # to update depending on the country
NB_OF_HH_PER_RES_BUILDING_RURAL = 1.0
CORRECTION_FACTOR_URBAN_HH_ACCESS = 1.0 # For HHwithAccess_urb calculation

# Residential energy per HH - Method 1 (Logistic RWI)
LOGISTIC_E_THRESHOLD = 4656 # kWh, adjust to country
LOGISTIC_ALPHA_DERIVATION_THRESHOLD = 0.1 # set so that E_HH = 7kWh for lowest tier
# alpha = E_threshold / 0.1 - 1. This implies a specific low energy value.
# To get E_HH = 7kWh when rwi_norm is low (exp term ~1), E_threshold / (1+alpha) = 7.
# For now, will keep E_threshold and how alpha is derived from it.
LOGISTIC_K_INITIAL_GUESS = 5.0

# DHS Data parameters
DHS_ELEC_KWH_ASSESSED_SURVEY = 'electricity_cons_kWH'
DHS_MAKE_FIGURE = True
DHS_RECALCULATE_ENERGIES = True
DHS_SIMULATE_CELL_GROUPS = True
DHS_RECALCULATE_ENERGY_PERHH = False
DHS_EMPLOYMENT_CATEGORIES = ['professional/technical/managerial', 'clerical', 'sales', 'services', 'skilled manual']
DHS_WORKING_AGE_GROUP_KEY = '15-49'


# Tiers for comparison
BINS_TIERS_ENERGY = [0, 7, 72.9-0.1, 364.9-0.1, 1250.4-0.1, 3012.2-0.1, float('inf')]

# Services demand parameters
# Weights for weighted average of services energy (alpha: GDP, beta: buildings, gamma: employees)
SERVICES_WEIGHT_GDP = 0.0 # alpha
SERVICES_WEIGHT_BUILDINGS = 0.0 # beta
SERVICES_WEIGHT_EMPLOYEES = 1.0 # gamma
# THRESHOLD_ACCESS_SERVICES = 0.1 # from original script, check if used with new structure

# Plotting parameters
MAP_DEFAULT_CMAP = "Reds"
MAP_LOG_NORM_VMIN = 1e-6

# -----------------------------------------------------------------------------
# COLUMN NAMES FOR GRID
# -----------------------------------------------------------------------------
# Input columns from H3 grid
COL_H3_ID = 'h3_index' # the ID column in h3_grid_at_hex.shp

# Processed column names
COL_BUILDINGS_SUM = 'buildingssum'
COL_LOCATION_WP = 'locationWP' # after processing_raster
COL_HREA_MEAN = 'HREA' # after processing_raster and rename
PROB_ELEC_COL = COL_HREA_MEAN # residential analysis parameters
COL_RWI_MEAN = 'rwi' # after processing_raster and rename
COL_TIERS_FALCHETTA_MAJ = 'tiers_falchetta_maj'
COL_TIERS_FALCHETTA_MEAN = 'tiers_falchetta_mean'
# COL_GDP_PPP_MEAN = 'GDP_PPP'

COL_ADMIN_NAME = ADMIN_REGION_COLUMN_NAME

COL_LOC_ASSESSED = 'location' # Final urban/rural column
COL_STATUS_ELECTRIFIED = 'status_electrified'
COL_IS_NEAR_ANY_LINE = 'is_near_any_line'

COL_RES_URBAN_BUI = 'res_urbanBui'
COL_RES_RURAL_BUI = 'res_ruralBui'
COL_RES_BUI = 'resBui'
COL_HH_URBAN = 'HH_urban'
COL_HH_RURAL = 'HH_rural'
COL_HH_TOTAL = 'HH_total'
COL_POPULATION = 'pop_total'
COL_POP_URBAN = 'pop_urban'
COL_POP_RURAL = 'pop_rural'

COL_HH_WITH_ACCESS_URB = 'HHwithAccess_urb'
COL_HH_WITH_ACCESS_RUR = 'HHwithAccess_rur'
COL_HH_WITH_ACCESS = 'HHwithAccess'
COL_HH_WO_ACCESS_URB = 'HHwoAccess_urb'
COL_HH_WO_ACCESS_RUR = 'HHwoAccess_rur'
COL_HH_WO_ACCESS = 'HHwoAccess'

COL_RWI_NORM = 'rwi_norm'
COL_RES_ELEC_PER_HH_LOG = 'elec_PerHH_kWh_log'
COL_RES_ELEC_PER_HH_DHS = 'elec_PerHH_kWh_DHS' # from estimate_energy script
COL_RES_ELEC_KWH_METH1 = 'elec_kWh_meth1'
COL_RES_ELEC_KWH_METH2 = 'elec_kWh_meth2'
COL_RES_ELEC_KWH_METH1_SCALED = 'resElec_kWh_meth1_scaled'
COL_RES_ELEC_KWH_METH2_SCALED = 'resElec_kWh_meth2_scaled'

COL_SER_BUI = 'serBui'
COL_SER_BUI_ACC = 'serBUi_access'
COL_SER_ELEC_KWH_BUI = 'Ser_elec_kWh_bui'
# COL_SER_ELEC_KWH_GDP = 'Ser_elec_kWh_GDP' # If GDP method is used
COL_TOTAL_EMPLOYEE = 'total_employee'
COL_TOTAL_EMPLOYEE_WITH_ACCESS = 'total_employee_withaccess'
COL_SER_ELEC_KWH_EMP = 'ser_elec_kWh_Emp'
COL_SER_ELEC_KWH_FINAL = 'ser_elec_kWh_final' # Final services result

# -----------------------------------------------------------------------------
# RESULTS FILES NAME
# -----------------------------------------------------------------------------
RESIDENTIAL_GRID_FILE = RESIDENTIAL_OUTPUT_DIR / f'data_res_{COUNTRY}.csv'
SERVICES_GRID_FILE = RESIDENTIAL_OUTPUT_DIR / f'data_ser_{COUNTRY}.csv'