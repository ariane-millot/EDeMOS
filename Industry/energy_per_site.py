###############################################################################                                                                                   
# Chair:            Chair of Renewable and Sustainable Energy Systems (ENS)
# Assistant(s):     Andjelka Kerekes (andelka.bujandric@tum.de)

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
    material = "Cu"
    #result = pd.ExcelWriter(os.path.join(path, "Energy_demands.xlsx"), engine='xlsxwriter')
    #----------------------------------
    # settings
    ## general
    plant_usage_ore = 1.0
    plant_usage_metal = 0.5
    ore_grade = 0.0062
    concentrate_grade = 0.3
    
    ## energy usage for different processes and their types
    ## Units: GJ/t_Cu
    spec_energy = {"Elec": {"Mining":{}, "Smelting":{}}, "Diesel":{"Mining":{}, "Smelting":{}}}
    #dict.fromkeys(["Elec", "Diesel"],{"Mining":{}, "Smelting":{}})#, "Refining") 
    ### mining
    spec_energy["Elec"]["Mining"]["Open Pit"]= 0
    spec_energy["Diesel"]["Mining"]["Open Pit"]= 10.2
    spec_energy["Elec"]["Mining"]["Underground"]= 2.15
    spec_energy["Diesel"]["Mining"]["Underground"]= 2.15
    mining_default = "Underground"
    spec_energy["Elec"]["Milling"] = 0.4*24 # weir engeco: Mining energy consumption 2021
    spec_energy["Diesel"]["Milling"]= 0
    
    ### smelting
    spec_energy["Elec"]["Smelting"]["Flash smelting"]= 9.266 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Diesel"]["Smelting"]["Flash smelting"]= 1.518 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Elec"]["Smelting"]["Isasmelt"]= 6.903 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Diesel"]["Smelting"]["Isasmelt"]= 4.175 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Elec"]["Smelting"]["Mitsubishi"]= 8.508 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Diesel"]["Smelting"]["Mitsubishi"]= 2.498 # https://link.springer.com/article/10.1007/s11837-015-1380-1
    spec_energy["Elec"]["Smelting"]["average"]= 0.4*8.9 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    spec_energy["Diesel"]["Smelting"]["average"]= 0.6*8.9 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    smelting_default = "Flash smelting"
    ### refining
    spec_energy["Elec"]["Refining"]= 3.2 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    spec_energy["Diesel"]["Refining"]= 0 # https://elib.dlr.de/130069/1/Renewable%20energy%20in%20copper%20production%20-%20a%20review.pdf
    
    #-----------------------------------
    
    output_table = input_table[["Country", "FeatureNam", "DsgAttr03", "DsgAttr06", "MemoOther", "MemoLoc","Latitude", "Longitude", "DsgAttr07"]].copy()
    
    # Create a list of all possible outputs and allocate Metal or Ore and concentrate to each process
    # clear output type
    output_type_list = []
    for idx, output_name in enumerate(input_table["DsgAttr03"]): # go through all commodity products
        if output_name not in ["Metal", "Ore and concentrate"]:
            if material+" content" in output_name:
                output_name = "Metal"
            elif "Ore." in output_table["MemoOther"][idx]:
                output_name = "Ore and concentrate"
            else:
                output_name = ""
        output_type_list.append(output_name)
    output_table["Output type (ass.)"] = output_type_list

    # Assess production level for each site
    # clear production
    production_2017 = []
    for idx, production in enumerate(input_table["DsgAttr07"]):
        if production <0:
            production = np.nan
            print("Production at a site ", output_table["FeatureNam"][idx], " in ", country, " is missing (negative). Value set to zero. Please, change the input in the input file.")
        elif "Capacity is a combination" in output_table["MemoOther"][idx]:
            number = output_table["MemoOther"].tolist().count(output_table["MemoOther"][idx]) # how many plants are giving the combined capacity
            production = production / number # in sum the plants have to give the value which is written next to each of them
        if output_table["Output type (ass.)"][idx]=="Metal":
            plant_usage = plant_usage_metal
        else:
            plant_usage = plant_usage_ore
        production_2017.append(production*plant_usage)
    output_table["Production (ass.) 2017 [t]"] = production_2017
    

    
    # Production of copper content in kt
    metal_content_list = []
    for idx, production in enumerate(output_table["Production (ass.) 2017 [t]"]):
        if output_table["Output type (ass.)"][idx]=="Ore and concentrate":
            metal_content_list.append(production*ore_grade/1000) # conversion in ktCu
        else:
            metal_content_list.append(production/1000)
    output_table["Cu content [kt]"] = metal_content_list
    
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
    for name in output_table["FeatureNam"]:
        if "smelter" in name.lower():
            metal_proces_list.append("Smelter")
        elif "refinery" in name.lower():
            metal_proces_list.append("Refinery")
        elif "plant" in name.lower() or "facility" in name.lower():
            metal_proces_list.append("Smelter+Refinery")
        else:
            metal_proces_list.append("")    
    output_table["Metal processing"] = metal_proces_list   
    
    # find the type of a smelter
    smelter_type_list = []
    types = ["Isasmelt","Mitsubishi", "Slag Recovery","Heap Leach","Leach","Solvent Extraction-Electrowinning", "Electrowinning", "Solvent Extraction"]
    for featureName in output_table["FeatureNam"]:
        element_type = [i for i in types if i.lower() in featureName.lower()]
        element_type = ", ".join(element for element in element_type)
        smelter_type_list.append(element_type)
    output_table["Smelter type"] = smelter_type_list   
    
    # attribute specific energy consumptions
    for en_carrier in spec_energy.keys():
        spec_energy_list = []
        for idx, element in enumerate(output_table["Output type (ass.)"]):
            # for ores consider ore grade
            if element == "Ore and concentrate":
                try:
                    spec_energy_list.append(spec_energy[en_carrier]["Mining"][output_table["Mine type"][idx]]*ore_grade
                                            +spec_energy[en_carrier]["Milling"]*ore_grade)
                except:
                    spec_energy_list.append(spec_energy[en_carrier]["Mining"][mining_default]*ore_grade
                                            +spec_energy[en_carrier]["Milling"]*ore_grade)
            
            elif element == "Metal":
                if output_table["Metal processing"][idx]=="Refinery":
                    spec_energy_list.append(spec_energy[en_carrier]["Refining"])
                elif output_table["Metal processing"][idx]=="Smelter+Refinery":
                    spec_energy_list.append(spec_energy[en_carrier]["Refining"]+spec_energy[en_carrier]["Smelting"]["Flash smelting"])
                else:
                    try:
                        spec_energy_list.append(spec_energy[en_carrier]["Smelting"][output_table["Smelter type"][idx]])
                    except:
                        spec_energy_list.append(spec_energy[en_carrier]["Smelting"][smelting_default])
    
            else:
                spec_energy_list.append("")
        output_table["Spec energy "+en_carrier+" [GJ/t]"]= spec_energy_list   
    
    # calculate consumed energy:  multiply production and specific energy consumption
    for en_carrier in spec_energy.keys():
        output_table["Energy " + en_carrier +" [TJ]"] = np.array(output_table["Production (ass.) 2017 [t]"])*np.array(output_table["Spec energy "+en_carrier+" [GJ/t]"])/1000
     
    ## Adding a uid which is needed afterwards
    output_table["id"] = range(1, len(output_table)+1)
    
    ## Export to csv
    output_table.to_csv(os.path.join(path_mines, display_name + ".csv"), encoding='utf-8', index=False)
    
    ## Converting df to gdf for further processing & extracting gpkg
    output_table_gdf = gpd.GeoDataFrame(output_table,
                                        geometry=gpd.points_from_xy(output_table.Longitude, output_table.Latitude), crs={'init': 'epsg:4326'})
    
    output_table_gdf.to_file(os.path.join(path_mines, ((display_name + ".gpkg"))), 
                                          layer=display_name, driver="GPKG")
    #output_table.to_excel(result, sheet_name=country, index=False, startrow=0)
    #result.close()
