from pyproj import CRS
from pathlib import Path

ROOT_DIR = Path( __file__ ).parent.absolute()

# -----------------------------------------------------------------------------
# AREA OF INTEREST CHOICE -- to update with the country
# -----------------------------------------------------------------------------
COUNTRY = "Kenya"
ISO_CODE = "KEN"
# Define area of interest
AREA_OF_INTEREST = "COUNTRY"  # Can be "COUNTRY" or a specific region like "Copperbelt"

YEAR = 2019 # year of analysis

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DATA_FOLDER = ROOT_DIR / "Data"
# Input paths
ADMIN_PATH = DATA_FOLDER / "admin"
ADMIN_GPKG = f"gadm41_{ISO_CODE}.gpkg"

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

# hexagon size
HEX_SIZE = 6 ## resolution info here https://h3geo.org/docs/core-library/restable

# HEXAGON FILE NAME
H3_GRID_HEX_SHP = f"h3_grid_at_hex_{COUNTRY}.shp" # Located in current OUTPUT_DIR

# BUFFER DISTANCE in meters.
buffer_distance_meters = 1000 # This should be larger than half the diagonal of a hexagon


# -----------------------------------------------------------------------------
# RESIDENTIAL PARAMETERS
# -----------------------------------------------------------------------------

# File paths

# Input paths
GRID_PATH = DATA_FOLDER / "Grid" / COUNTRY # For MV/HV lines
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
WP_BUILDINGS_COUNT_TIF = f"{ISO_CODE}_buildings_v2_0_count.tif"
WP_BUILDINGS_URBAN_TIF = f"{ISO_CODE}_buildings_v2_0_urban.tif"

# Lighting file
# # set_lightscore_sy_xxxx.tif: Predicted likelihood that a settlement is electrified (0 to 1)
# Link: http://www-personal.umich.edu/~brianmin/HREA/data.html
HREA_LIGHTING_TIF = f"{COUNTRY}_set_lightscore_sy_2019.tif"

# RWI file
# Link: https://gee-community-catalog.org/projects/rwi/
# run the notebook Rasterize RWI.ipynb to generate rwi_map.tif
RWI_MAP_TIF = f"rwi_map_{COUNTRY}.tif"
RWI_FILE_CSV = f"{ISO_CODE.lower()}_relative_wealth_index.csv"

# Falchetta Tiers file
# run the script transform-nc_to_tiff.py to generate .tif file
FALCHETTA_TIERS_TIF = f"tiersofaccess_2018_band1_{COUNTRY}.tif"

# GDP file Kummu dataset
# Link https://www.nature.com/articles/sdata20184#Sec9
# GDP_PPP_TIF = "GDP_PPP_30arcsec_v3_band3_Zambia.tif"

# UN Energy Balance file
# https://data.un.org/SdmxBrowser/start
UN_ENERGY_BALANCE_CSV = f"UNSD+DF_UNData_EnergyBalance+1.0_{COUNTRY}.csv"

# Grid line files
# Data available at https://datacatalog.worldbank.org/search/dataset/0040190/Zambia---Electricity-Transmission-Network
# or https://energydata.info/dataset/zambia-electrical-lines
MV_LINES_SHP = GRID_PATH / COUNTRY / "Transmission lines 132kV" / "132kV.shp"
HV_LINES_SHP = GRID_PATH / COUNTRY /"Transmission lines 220kV" / "220kV.shp"

# Census data files
# The file should contain the following data, region, HH urban, rural, total, size of HH urban/rural
PROVINCE_DATA_AVAILABLE = True
CENSUS_PROVINCE_CSV = RESIDENTIAL_DATA_PATH / "Census" / COUNTRY /"Census_KEN.csv"
CENSUS_NATIONAL_CSV = RESIDENTIAL_DATA_PATH / "Census" / COUNTRY / "Census_KEN_National.csv"

# DHS Survey related files (used by estimate_energy_rwi_link_national_new.py)
# 1. Run read_DHS_hh_to_df.py in HouseholdEnergyUse folder to generate the household_data.csv
# 2. Run estimate_energy_perhh_DHS.py in HouseholdEnergyUse folder
# For services:
# 1 Download the Individual Recode  XXXXDT.ZIP to generate employee_survey_(men/women).csv
# 2 Run: read_DHS_services_to_df.py in HouseholdEnergyUse folder
# 3 Download or add the 'pop15-49_share_{COUNTRY}.csv' file in the DHS folder
DHS_FOLDER = RESIDENTIAL_DATA_PATH / "DHS" / COUNTRY
DHS_HOUSEHOLD_DATA_CSV = DHS_FOLDER / f'household_data_{COUNTRY}.csv'
DHS_EMPLOYEE_WOMEN_CSV = DHS_FOLDER / f'employee_survey_women_{COUNTRY}.csv'
DHS_EMPLOYEE_MEN_CSV = DHS_FOLDER / f'employee_survey_men_{COUNTRY}.csv'
DHS_WORKING_POP_SHARE_CSV = DHS_FOLDER / f'pop15-49_share_{COUNTRY}.csv'

# Ensure all folders for output files exist
RESIDENTIAL_OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DHS_FOLDER.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# INDUSTRY PARAMETERS
# -----------------------------------------------------------------------------
MINES_DATA_PATH = ROOT_DIR / "Industry/Data/mines"
MINES_INPUT_CSV = MINES_DATA_PATH / "Mineral_Facilities_correctedInput.csv"
MINES_OUTPUT_GPKG = MINES_DATA_PATH / f"mineral_facilities_{COUNTRY.lower()}.gpkg"
MINES_OUTPUT_CSV = MINES_DATA_PATH / f"mineral_facilities_{COUNTRY.lower()}.csv"
COL_IND_ELEC_TJ = "ind_elec_TJ"
COL_IND_ELEC_GWH = "ind_elec_GWh"
COL_IND_ELEC_KWH = "ind_elec_kWh"
COL_IND_OIL_TJ = "ind_diesel_TJ"
COL_IND_TOTAL_TJ = "ind_total_energy_TJ"
COL_IND_COPPER_ELEC_TJ = "copper_elec_TJ"

INDUSTRY_OUTPUT_DIR = ROOT_DIR / "Industry/Outputs"
INDUSTRY_OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# DHS FILES PARAMETERS
# -----------------------------------------------------------------------------
DHS_HH_SURVEY_FILE = 'KEHR8CFL'
DHS_SERVICES_SURVEY_FILE = 'KEIR8CFL'

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

# UN Energy Balance year
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
# RESULTS FILES NAME
# -----------------------------------------------------------------------------
RESIDENTIAL_GRID_FILE = RESIDENTIAL_OUTPUT_DIR / f'data_res_{COUNTRY}.csv'
SERVICES_GRID_FILE = RESIDENTIAL_OUTPUT_DIR / f'data_ser_{COUNTRY}.csv'
BUILDINGS_GPKG_FILE = RESIDENTIAL_OUTPUT_DIR / f'buildings_map_{COUNTRY}.gpkg'
INDUSTRY_GPKG_FILE = INDUSTRY_OUTPUT_DIR / f'ind_energy_map_{COUNTRY}.gpkg'
TOTAL_ELECTRICITY_GPKG_FILE = OUTPUT_DIR / f'total_electricity_consumption_{COUNTRY}.gpkg'