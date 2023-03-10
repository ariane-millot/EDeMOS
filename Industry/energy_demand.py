import numpy as np
from pandas import read_csv
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.tile_providers import get_provider, ESRI_IMAGERY, OSM
from bokeh.models import Panel, Tabs

data_folder = './Industry/Data/'

ElecNonFerrousMetals = 22897  # year 2019
ElecMining = 818  # year 2019

# Functions for converting degrees to web mercator x and y
RE = 6378137


def lat2y(a):
    return np.log(np.tan(np.pi/4 + np.radians(a)/2)) * RE


def lon2x(a):
    return np.radians(a) * RE


# Read in locations of mines from csv file
mine_data = read_csv(data_folder + 'mines_zambia.csv', header=6, thousands=',')
location = np.array(mine_data[['Latitude (degrees)', 'Longitude (degrees)']])

# Put the relevant data in a source dictionary which the plot will read from
mine_data_dict = dict(
    x=lon2x(location[:, 1]),
    y=lat2y(location[:, 0]),
    mine_name=mine_data['Property Name'],
    ore_processed=mine_data['Ore Processed Mass (tonnes)'],
    primary_commodity=mine_data['Primary Commodity']
  )

total_production = mine_data_dict['ore_processed'].sum(skipna=True)
# total_production = np.nansum(mine_data_dict['ore_processed'])
# print(total_production)
ElecNonFerrousMetalsPerMine = ElecNonFerrousMetals * mine_data_dict['ore_processed'] / total_production
print(ElecNonFerrousMetalsPerMine)
mine_data_dict['elec_mine'] = ElecNonFerrousMetalsPerMine

mine_data_source = ColumnDataSource(mine_data_dict)

# Define range bounds in web mercator coordinates
loc_range = np.array([np.amin(location, axis=0), np.amax(location, axis=0)])

# Set what is displayed when mouse hovers over point
TOOLTIPS = [
 ("Mine", "@mine_name"),
 ("Primary commodity", "@primary_commodity"),
 ("Ore processed", "@ore_processed tonnes/yr"),
 ("Electricity", "@elec_mine PJ")
]

# Set output filename and title
output_file("energy_demand_zambia.html", title='Estimated energy demand in Zambia')

figs = []
tabs = []
# Make two panels, one with open street map background, one with satellite
for panel_name, tile in [['Street Map', get_provider(OSM)], ['Satellite', get_provider(ESRI_IMAGERY)]]:
    # Set up figure pane and parameters
    p = figure(x_range=lon2x(loc_range[:, 1]), y_range=lat2y(loc_range[:, 0]),
               x_axis_type="mercator", y_axis_type="mercator",
               plot_width=1000, plot_height=700, tooltips=TOOLTIPS,
               tools="pan,wheel_zoom,reset,hover", active_scroll="wheel_zoom")
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.add_tile(tile)
    # Display mine sites as points
    p.circle('x', 'y', source=mine_data_source, fill_color='orange', line_color='red', fill_alpha=0.2, size=10)
    # Add figure to tabs
    tab = Panel(child=p, title=panel_name)
    tabs.append(tab)
    figs.append(p)

# Load output
show(Tabs(tabs=tabs))
