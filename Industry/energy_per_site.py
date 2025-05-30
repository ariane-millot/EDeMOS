###############################################################################                                                                                   
# Chair:            Chair of Renewable and Sustainable Energy Systems (ENS)
# Assistant(s):     Andjelka Kerekes (andelka.kerekes@tum.de)

# Date:             
# Version:          v3.0
# Status:           done/in progress
# Python-Version:   3.7.3 (64-bit)
###############################################################################
import pandas as pd
import os
import numpy as np
from copy import copy
import geopandas as gpd
import fiona

def calc_energy_per_site(path_mines, input_table, display_name):
  
    country = "Zambia"
    metals = ["Copper", "Cobalt", "Gold", "Nickel", "Manganese"]
    electrowon_names = ["Slag Recovery","Heap Leach","Leach","Solvent Extraction-Electrowinning", "Electrowinning", "Solvent Extraction"]
    
    input_table = input_table[input_table["Country"]==country]
    input_table = input_table.loc[input_table["DsgAttr02"].isin(metals)].reset_index()
    
    #----------------------------------
    # settings
    ## general
    plant_usage_ore = {"Copper": 0.773, "Cobalt":0, "Gold":0.853, "Nickel":0.285, "Manganese":0.942} # Co from mining not considered
    plant_usage_metal = {"Copper":  0.312, "Cobalt":0.136, "Gold":0.853, "Nickel":0, "Manganese":0.942} # no Ni metal production, no Au metal statistics, Mn ore and fero- or siliconmanganese statistics
    ore_grade = {"Copper":0.0103, "Cobalt":0} # cobalt from mining not considered
    concentrate_grade = 0.3
    unit_conversion = {"Thousand metric tons":10**3, "Metric tons":1, "Kilograms":10**-3} # into t
    
    ## energy usage for different processes and their types
    ## Units: GJ/t_Cu
    spec_energy = {"Copper": {"Elec": {"Mining":{}, "Smelting":{}}, "Diesel":{"Mining":{}, "Smelting":{}}},
                   "Cobalt":{"Elec":{"Mining":{}}, "Diesel":{"Mining":{}}},
                   "Gold":{"Elec":{"Mining":{}}, "Diesel":{"Mining":{}}},
                   "Nickel":{"Elec":{"Mining":{}}, "Diesel":{"Mining":{}}},
                   "Manganese":{"Elec":{"Mining":{}}, "Diesel":{"Mining":{}}}}
    
    ### mining
    spec_energy["Copper"]["Elec"]["Mining"]["Open Pit"]= 0        # https://doi.org/10.1016/j.jclepro.2019.118978
    spec_energy["Copper"]["Diesel"]["Mining"]["Open Pit"]= 10.2   # https://doi.org/10.1016/j.jclepro.2019.118978
    spec_energy["Copper"]["Elec"]["Mining"]["Underground"]= 2.15  # https://doi.org/10.1016/j.jclepro.2019.118978
    spec_energy["Copper"]["Diesel"]["Mining"]["Underground"]= 2.15 # https://doi.org/10.1016/j.jclepro.2019.118978
    mining_default = "Underground"
    spec_energy["Copper"]["Elec"]["Milling"] = 0.4*24 # weir engeco: Mining energy consumption 2021
    spec_energy["Copper"]["Diesel"]["Milling"]= 0     # weir engeco: Mining energy consumption 2021
    
    spec_energy["Cobalt"]["Elec"]["Mining"][mining_default]= np.nan # placeholder
    spec_energy["Cobalt"]["Diesel"]["Mining"][mining_default]= np.nan # placeholder
    spec_energy["Cobalt"]["Elec"]["Milling"]= np.nan # placeholder
    spec_energy["Cobalt"]["Diesel"]["Milling"]= np.nan # placeholder
    
    spec_energy["Gold"]["Elec"]["Mining"][mining_default]= (24.11*10**3)/2
    spec_energy["Gold"]["Diesel"]["Mining"][mining_default]=(223.21+36.21)*10**3/2
    spec_energy["Gold"]["Elec"]["Milling"]= (96.71+34.81)*10**3/2 
    spec_energy["Gold"]["Diesel"]["Milling"]= 0
    
    spec_energy["Nickel"]["Elec"]["Mining"][mining_default]= 18       
    spec_energy["Nickel"]["Diesel"]["Mining"][mining_default]=12.5
    spec_energy["Nickel"]["Elec"]["Milling"]= 0       #Mining+Milling one value
    spec_energy["Nickel"]["Diesel"]["Milling"]= 0  #Mining+Milling one value
    
    spec_energy["Manganese"]["Elec"]["Mining"][mining_default]= 0.025     
    spec_energy["Manganese"]["Diesel"]["Mining"][mining_default]= 0.17
    spec_energy["Manganese"]["Elec"]["Milling"]= 0    #Mining+Milling one value
    spec_energy["Manganese"]["Diesel"]["Milling"]= 0  #Mining+Milling one value
    
    ### smelting
    spec_energy["Copper"]["Elec"]["Smelting"]["Flash smelting"]= 9.266 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Diesel"]["Smelting"]["Flash smelting"]= 1.518 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Elec"]["Smelting"]["Isasmelt"]= 6.903 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Diesel"]["Smelting"]["Isasmelt"]= 4.175 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Elec"]["Smelting"]["Mitsubishi"]= 8.508 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Diesel"]["Smelting"]["Mitsubishi"]= 2.498 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Copper"]["Elec"]["Smelting"]["average"]= 0.4*8.9 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    spec_energy["Copper"]["Diesel"]["Smelting"]["average"]= 0.6*8.9 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    smelting_default = "Flash smelting"
    ### refining
    spec_energy["Copper"]["Elec"]["Refining"]= 3.2*0.4 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    spec_energy["Copper"]["Diesel"]["Refining"]= 3.2*0.6 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    
    ### leaching, solvent extraction and electrowinning
    spec_energy["Copper"]["Elec"]["Hydrometallurgical"]= 14.7*0.85 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    spec_energy["Copper"]["Diesel"]["Hydrometallurgical"]= 14.7*0.15 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    
    ### smelting, refining = processing
    spec_energy["Cobalt"]["Elec"]["Smelting/Refining"]= 13.57
    spec_energy["Cobalt"]["Diesel"]["Smelting/Refining"]= 0
    
    spec_energy["Gold"]["Elec"]["Smelting/Refining"]= (52.08+38.86)*10**3/2 # Asuming total energy = elec
    spec_energy["Gold"]["Diesel"]["Smelting/Refining"]= 0
    
    spec_energy["Nickel"]["Elec"]["Smelting/Refining"]= 200 # Asuming total energy = elec
    spec_energy["Nickel"]["Diesel"]["Smelting/Refining"]= 0
    
    spec_energy["Manganese"]["Elec"]["Smelting/Refining"]= 18.01
    spec_energy["Manganese"]["Diesel"]["Smelting/Refining"]= 13.02
    
    #-----------------------------------
    
    output_table = input_table[["Country", "FeatureNam", "DsgAttr02", "DsgAttr03", "DsgAttr06", "MemoOther", "MemoLoc","Latitude", "Longitude", "DsgAttr07", "DsgAttr08"]].copy()
    
    # Create a list of all possible outputs and allocate Metal or Ore and concentrate to each process
    # clear output type
    output_type_list = []
    for idx, output_name in enumerate(input_table["DsgAttr03"]): # go through all commodity products
        metal = input_table["DsgAttr02"][idx]
        
        if output_name not in ["Metal", "Ore and concentrate"]:
            if " content" in output_name:
                output_name = "Metal in ore"
            elif "Ore." in output_table["MemoOther"][idx]:
                output_name = "Ore and concentrate"
            elif output_name == metal:
                output_name = "Metal" # "Metal" or "Ore and concentrate" - discutable
            else:
                output_name = ""
        output_type_list.append(output_name)
    output_table["Output type (ass.)"] = output_type_list
    
    # Assess production level for each site
    # clear production
    production_2017 = []
    for idx, production in enumerate(input_table["DsgAttr07"]):
        metal = input_table["DsgAttr02"][idx]
        unit_orig = input_table["DsgAttr08"][idx]
        
        if production <0:
            production = np.nan
            print("Production at a site ", output_table["FeatureNam"][idx], " in ", country, " is missing (negative). Value set to zero. Please, change the input in the input file.")
        elif "Capacity is a combination" in output_table["MemoOther"][idx]:
            number = output_table["MemoOther"].tolist().count(output_table["MemoOther"][idx]) # how many plants are giving the combined capacity
            production = production / number 
        if output_table["Output type (ass.)"][idx]=="Metal":
            plant_usage = plant_usage_metal[metal]
        else:
            plant_usage = plant_usage_ore[metal]
        try:
            production_2017.append(production*plant_usage*unit_conversion[unit_orig]) 
        except:
            production_2017.append(production*plant_usage) # original unit is sometimes nan
    output_table["Production (ass.) 2019 [t]"] = production_2017
      
    # Production of copper content in kt
    metal_content_list = []
    for idx, production in enumerate(output_table["Production (ass.) 2019 [t]"]):
        metal = input_table["DsgAttr02"][idx]
        
        if output_table["Output type (ass.)"][idx]=="Ore and concentrate":
            metal_content_list.append(production*ore_grade[metal]/1000) # conversion in kt metal
        else:
            metal_content_list.append(production/1000)
    output_table["Metal content [kt]"] = metal_content_list
    
    # find mine type
    mine_type_list = []
    for name in output_table["FeatureNam"]:
        if "underground" in name.lower():
            mine_type_list.append("Underground")
        elif "pit" in name.lower():
            mine_type_list.append("Open Pit")
        else:
            mine_type_list.append("")
    output_table["Mine type"] = mine_type_list
    
    # find if site is a smelter, refinery or a combination
    metal_proces_list = []
    for idx, name in enumerate(output_table["FeatureNam"]):
        metal = input_table["DsgAttr02"][idx]
        
        if metal == "Copper":
            if "smelter" in name.lower():
                metal_proces_list.append("Smelter")
            elif "refinery" in name.lower():
                metal_proces_list.append("Refinery")
            elif any(x.lower() in name.lower() for x in electrowon_names):
                metal_proces_list.append("Hydrometallurgical")
            elif "plant" in name.lower() or "facility" in name.lower():
                metal_proces_list.append("Smelter+Refinery")
            else:
                metal_proces_list.append("")
        else:
            if output_table["Output type (ass.)"][idx]=="Metal":
                metal_proces_list.append("Smelting/Refining")
            else:
                metal_proces_list.append("")    
    output_table["Metal processing"] = metal_proces_list   
    
    # find the type of a smelter
    smelter_type_list = []
    types = ["Isasmelt","Mitsubishi"] + electrowon_names
    for featureName in output_table["FeatureNam"]:
        element_type = [i for i in types if i.lower() in featureName.lower()]
        element_type = ", ".join(element for element in element_type)
        smelter_type_list.append(element_type)
    output_table["Metal process type"] = smelter_type_list   
    
    # attribute specific energy consumptions
    for en_carrier in spec_energy["Copper"].keys():
        spec_energy_list = []
        for idx, element in enumerate(output_table["Output type (ass.)"]):
            
            metal = input_table["DsgAttr02"][idx]
            if element in ["Ore and concentrate", "Metal in ore"]:
                try:
                    spec_energy_list.append(spec_energy[metal][en_carrier]["Mining"][output_table["Mine type"][idx]]
                                            +spec_energy[metal][en_carrier]["Milling"])
                except:
                    spec_energy_list.append(spec_energy[metal][en_carrier]["Mining"][mining_default]
                                            +spec_energy[metal][en_carrier]["Milling"])
            
            elif element == "Metal":
                if output_table["Metal processing"][idx] in ["Refining","Hydrometallurgical","Smelting/Refining"]:
                    spec_energy_list.append(spec_energy[metal][en_carrier][output_table["Metal processing"][idx]])
                elif output_table["Metal processing"][idx]=="Smelter+Refinery":
                    spec_energy_list.append(spec_energy[metal][en_carrier]["Refining"]+spec_energy[metal][en_carrier]["Smelting"]["Flash smelting"])
                else:
                    try:
                        spec_energy_list.append(spec_energy[metal][en_carrier]["Smelting"][output_table["Metal process type"][idx]])
                    except:
                        spec_energy_list.append(spec_energy[metal][en_carrier]["Smelting"][smelting_default])
    
            else:
                spec_energy_list.append(0) #""
        output_table["Spec energy "+en_carrier+" [GJ/t]"]= spec_energy_list   
    
    # calculate consumed energy:  multiply production and specific energy consumption
    for en_carrier in spec_energy["Copper"].keys():
        output_table[en_carrier +"_TJ"] = np.array(output_table["Metal content [kt]"])*np.array(output_table["Spec energy "+en_carrier+" [GJ/t]"])

    # ---- ADDITION FOR COPPER-SPECIFIC ELECTRICITY CONSUMPTION ----
    # Initialize the new column with 0
    output_table["Copper_Elec_Cons_TJ"] = 0.0

    # Create a boolean mask for rows where the metal is Copper
    is_copper_mask = (output_table["DsgAttr02"] == "Copper")

    # For copper rows, calculate electricity consumption.
    output_table["Copper_Elec_Cons_TJ"] = np.where(
        output_table["DsgAttr02"] == "Copper",  # Condition
        output_table["Metal content [kt]"] * output_table["Spec energy Elec [GJ/t]"], # Value if True
        0.0  # Value if False
    )
     
    ## Adding a uid which is needed afterwards
    output_table["id"] = range(1, len(output_table)+1)
    
    ## Export to csv
    output_table.to_csv(os.path.join(path_mines, display_name + ".csv"), encoding='utf-8', index=False)
    
    ## Converting df to gdf for further processing & extracting gpkg
    output_table_gdf = gpd.GeoDataFrame(output_table,
                                        geometry=gpd.points_from_xy(output_table.Longitude, output_table.Latitude), crs='EPSG:4326')
    
    output_table_gdf.to_file(os.path.join(path_mines, ((display_name + ".gpkg"))), 
                                          layer=display_name, driver="GPKG")
