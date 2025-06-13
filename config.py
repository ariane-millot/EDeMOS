from pyproj import CRS
from pathlib import Path

ROOT_DIR = Path( __file__ ).parent.absolute()

# -----------------------------------------------------------------------------
# AREA OF INTEREST CHOICE -- to update with the country
# -----------------------------------------------------------------------------

# Define area of interest
AREA_OF_INTEREST = "COUNTRY"  # Can be "COUNTRY" or a specific region like "Copperbelt"
ADMIN_GPKG = "gadm41_ZMB.gpkg"


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

# Input paths
ADMIN_PATH = ROOT_DIR / "admin"

# Output paths
OUTPUT_DIR = ROOT_DIR / "Outputs"
# Ensure all folders for output files exist
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# COORDINATE REFERENCE SYSTEMS
# -----------------------------------------------------------------------------
CRS_WGS84 = CRS("EPSG:4326") # Original WGS84 coordinate system
CRS_PROJ = CRS("EPSG:32736") # Projection system for the selected country - see http://epsg.io/ for more info

# -----------------------------------------------------------------------------
# GENERAL PARAMETERS - HEXAGONS
# -----------------------------------------------------------------------------

# Admin boundaries
ADMIN_LAYER_REGION = "ADM_ADM_1"
ADMIN_LAYER_COUNTRY = "ADM_ADM_0"
ADMIN_REGION_COLUMN_NAME = "NAME_1"

# hexagon size
HEX_SIZE = 5 ## resolution info here https://h3geo.org/docs/core-library/restable

# HEXAGON FILE NAME
H3_GRID_HEX_SHP = "h3_grid_at_hex.shp" # Located in current OUTPUT_DIR

# -----------------------------------------------------------------------------
# RESIDENTIAL PARAMETERS
# -----------------------------------------------------------------------------

# File paths

# Input paths
GRID_PATH = ROOT_DIR / "Grid" # For MV/HV lines
ENERGY_BALANCE_PATH = ROOT_DIR / "EnergyBalance"
RESIDENTIAL_DATA_PATH = ROOT_DIR / "Residential" / "Data"
WORLDPOP_PATH = RESIDENTIAL_DATA_PATH / "WorldPop"
LIGHTING_PATH = RESIDENTIAL_DATA_PATH / "Lighting"
RWI_PATH = RESIDENTIAL_DATA_PATH / "WealthIndex"
FALCHETTA_PATH = RESIDENTIAL_DATA_PATH / "Falchetta_ElecAccess"
# GDP_PATH = RESIDENTIAL_DATA_PATH / "GDP") # Example if it was used

# Output paths
RESIDENTIAL_OUTPUT_DIR = ROOT_DIR / "Residential" / "Outputs" # As used in building_demand.py for dataHH_region.csv
FIGURES_DHS_FOLDER = ROOT_DIR / "Residential" / "Figures" # As used in estimate_energy_rwi_link_national_new.py

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
RWI_MAP_TIF = "rwi_map.tif"

# Falchetta Tiers file
FALCHETTA_TIERS_TIF = "Zambia_tiersofaccess_2018.tif"

# GDP file Kummu dataset
# Link https://www.nature.com/articles/sdata20184#Sec9
# GDP_PPP_TIF = "GDP_PPP_30arcsec_v3_band3_Zambia.tif"

# UN Energy Balance file
UN_ENERGY_BALANCE_CSV = "UNSD+DF_UNData_EnergyBalance+1.0_Zambia.csv"

# Grid line files
# Data available at https://datacatalog.worldbank.org/search/dataset/0040190/Zambia---Electricity-Transmission-Network
# or https://energydata.info/dataset/zambia-electrical-lines
MV_LINES_SHP = GRID_PATH / "Zambia - MVLines" / "Zambia - MVLines.shp"
HV_LINES_SHP = GRID_PATH / "Zambia - HVLines" / "HVLines.shp"

# Census data files
PROVINCE_DATA_AVAILABLE = True
CENSUS_ZAMBIA_PROVINCE_CSV = RESIDENTIAL_DATA_PATH / "Census" / "Census_Zambia.csv"
CENSUS_ZAMBIA_NATIONAL_CSV = RESIDENTIAL_DATA_PATH / "Census" /  "Census_Zambia_National.csv"

# DHS Survey related files (used by estimate_energy_rwi_link_national_new.py)
DHS_HOUSEHOLD_DATA_CSV = RESIDENTIAL_DATA_PATH / "DHSSurvey" / "household_data.csv"
DHS_EMPLOYEE_WOMEN_CSV = RESIDENTIAL_DATA_PATH / "DHSSurvey" / "employee_survey_women.csv"
DHS_EMPLOYEE_MEN_CSV = RESIDENTIAL_DATA_PATH / "DHSSurvey" / "employee_survey_men.csv"
DHS_WORKING_POP_SHARE_CSV = RESIDENTIAL_DATA_PATH / "DHSSurvey" / "pop15-49_share.csv"

# Ensure all folders for output files exist
RESIDENTIAL_OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DHS_FOLDER.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# COORDINATE REFERENCE SYSTEMS
# -----------------------------------------------------------------------------
# CRS_WGS84 = CRS("EPSG:4326")
# CRS_PROJ = CRS("EPSG:32736") # Zambia UTM Zone 36S
TARGET_CRS_METERS = "EPSG:32735" # Used for grid line buffering (UTM Zone 35S) - check if this is consistent or should be same as CRS_PROJ

# -----------------------------------------------------------------------------
# PARAMETERS ENERGY BALANCE
# -----------------------------------------------------------------------------


# UN Energy Balance codes and year
UN_ELEC_CODE = "B07_EL"
UN_HH_TRANSACTION_CODE = "B50_1231"
UN_SERVICES_TRANSACTION_CODE = "B49_1235"
UN_OTHER_TRANSACTION_CODE = "B51_1234" # Other consumption not elsewhere specified
UN_ENERGY_YEAR = 2019

# -----------------------------------------------------------------------------
# COLUMN NAMES FOR GRID
# -----------------------------------------------------------------------------
# Input columns from H3 grid
COL_H3_ID = 'h3_index' # Or whatever the ID column in h3_grid_at_hex.shp is, e.g. 'id'

# Processed column names (examples, expand as needed)
COL_BUILDINGS_SUM = 'buildingssum'
COL_LOCATION_WP = 'locationWP' # after processing_raster
COL_HREA_MEAN = 'HREA' # after processing_raster and rename
COL_RWI_MEAN = 'rwi' # after processing_raster and rename
COL_TIERS_FALCHETTA_MAJ = 'tiers_falchetta_maj'
COL_TIERS_FALCHETTA_MEAN = 'tiers_falchetta_mean'
# COL_GDP_PPP_MEAN = 'GDP_PPP'

COL_ADMIN_NAME = ADMIN_REGION_COLUMN_NAME

COL_LOC_ASSESSED = 'location' # Final urban/rural column
COL_STATUS_ELECTRIFIED = 'Status_electrified'
COL_IS_NEAR_ANY_LINE = 'is_near_any_line'

COL_RES_URBAN_BUI = 'res_urbanBui'
COL_RES_RURAL_BUI = 'res_ruralBui'
COL_RES_BUI = 'res_Bui'
COL_HH_URBAN = 'HH_urban'
COL_HH_RURAL = 'HH_rural'
COL_HH_TOTAL = 'HH_total'
COL_POPULATION = 'population'
COL_POP_URBAN = 'pop_urban'
COL_POP_RURAL = 'pop_rural'

COL_HH_WITH_ACCESS_URB = 'HHwithAccess_urb'
COL_HH_WITH_ACCESS_RUR = 'HHwithAccess_rur'
COL_HH_WITH_ACCESS = 'HHwithAccess'
COL_HH_WO_ACCESS_URB = 'HHwoAccess_urb'
COL_HH_WO_ACCESS_RUR = 'HHwoAccess_rur'
COL_HH_WO_ACCESS = 'HHwoAccess'

COL_RWI_NORM = 'rwi_norm'
COL_RES_ELEC_PER_HH_LOG = 'Elec_PerHH_kWh_log'
COL_RES_ELEC_PER_HH_DHS = 'Elec_PerHH_kWh_DHS' # from estimate_energy script
COL_RES_ELEC_KWH_METH1 = 'Elec_kWh_meth1'
COL_RES_ELEC_KWH_METH2 = 'Elec_kWh_meth2'
COL_RES_ELEC_KWH_METH1_SCALED = 'ResElec_kWh_meth1_scaled'
COL_RES_ELEC_KWH_METH2_SCALED = 'ResElec_kWh_meth2_scaled' # Final residential result often used for map

COL_SER_BUI = 'serBui'
COL_SER_BUI_ACC = 'serBUi_Acc'
COL_SER_ELEC_KWH_BUI = 'Ser_elec_kWh_bui'
# COL_SER_ELEC_KWH_GDP = 'Ser_elec_kWh_GDP' # If GDP method is used
COL_TOTAL_EMPLOYEE = 'total_employee'
COL_TOTAL_EMPLOYEE_WITH_ACCESS = 'total_employee_withaccess'
COL_SER_ELEC_KWH_EMP = 'Ser_elec_kWh_Emp'
COL_SER_ELEC_KWH_FINAL = 'Ser_elec_kWh_final' # Final services result


# -----------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# -----------------------------------------------------------------------------

# Residential demand parameters

PROB_ELEC_COL = COL_HREA_MEAN
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

# Residential energy per HH - Method 2 (DHS based)
# Parameters for estimate_energy_rwi_link_national_new.py itself
DHS_MAKE_FIGURE = True
DHS_RECALCULATE_ENERGIES = True
DHS_SIMULATE_CELL_GROUPS = True
DHS_RECALCULATE_ENERGY_PERHH = False # This likely triggers a sub-script

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