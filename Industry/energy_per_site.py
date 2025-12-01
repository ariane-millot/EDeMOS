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
import geopandas as gpd
import fiona
import config
import specified_energy

def calc_energy_per_site(app_config):

    #----------------------------------
    # settings
    country = app_config.COUNTRY
    metals = ["Copper", "Cobalt", "Gold", "Nickel", "Manganese"]
    electrowon_names = ["Slag Recovery","Heap Leach","Leach","Solvent Extraction-Electrowinning", "Electrowinning", "Solvent Extraction"]
    ## general
    plant_usage_ore = {"Copper": 0.773, "Cobalt":0, "Gold":0.853, "Nickel":0.285, "Manganese":0.942} # Co from mining not considered
    plant_usage_metal = {"Copper":  0.312, "Cobalt":0.136, "Gold":0.853, "Nickel":0, "Manganese":0.942} # no Ni metal production, no Au metal statistics, Mn ore and fero- or siliconmanganese statistics
    ore_grade = {"Copper":0.0103, "Cobalt":0} # cobalt from mining not considered
    concentrate_grade = 0.3
    unit_conversion = {"Thousand metric tons":10**3, "Metric tons":1, "Kilograms":10**-3} # into t
    spec_energy = specified_energy.SPEC_ENERGY
    mining_default = specified_energy.MINING_DEFAULT
    smelting_default = specified_energy.SMELTING_DEFAULT

    #----------------------------------

    ## Load USGS data
    input_table = pd.read_csv(app_config.MINES_INPUT_CSV)
    input_table = input_table[input_table["Country"]==country]
    input_table = input_table.loc[input_table["DsgAttr02"].isin(metals)].reset_index()
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
    # Rename the columns
    output_table = output_table.rename(columns={
        "Elec_TJ": app_config.COL_IND_ELEC_TJ,
        "Diesel_TJ": app_config.COL_IND_OIL_TJ
    })

    # ---- ADDITION FOR COPPER-SPECIFIC ELECTRICITY CONSUMPTION ----
    # Initialize the new column with 0
    output_table[app_config.COL_IND_COPPER_ELEC_TJ] = 0.0

    # Create a boolean mask for rows where the metal is Copper
    is_copper_mask = (output_table["DsgAttr02"] == "Copper")

    # For copper rows, calculate electricity consumption.
    output_table[app_config.COL_IND_COPPER_ELEC_TJ] = np.where(
        output_table["DsgAttr02"] == "Copper",  # Condition
        output_table["Metal content [kt]"] * output_table["Spec energy Elec [GJ/t]"], # Value if True
        0.0  # Value if False
    )
     
    ## Adding a uid which is needed afterwards
    output_table["id"] = range(1, len(output_table)+1)
    
    ## Export to csv
    output_table.to_csv(app_config.MINES_OUTPUT_CSV, encoding='utf-8', index=False)
    
    ## Converting df to gdf for further processing & extracting gpkg
    output_table_gdf = gpd.GeoDataFrame(output_table,
                                        geometry=gpd.points_from_xy(output_table.Longitude, output_table.Latitude), crs=app_config.CRS_WGS84)
    if os.path.exists(app_config.MINES_OUTPUT_GPKG):
        print(f"File {app_config.MINES_OUTPUT_GPKG} exists. Deleting it now...")
        os.remove(app_config.MINES_OUTPUT_GPKG)
        print("File deleted.")
    output_table_gdf.to_file(app_config.MINES_OUTPUT_GPKG, layer="mines", driver="GPKG", mode='w')
    # try:
    #     layer_names = fiona.listlayers(app_config.MINES_OUTPUT_GPKG)
    #     print(f"Layers found in '{app_config.MINES_OUTPUT_GPKG}':")
    #     for name in layer_names:
    #         print(f"- {name}")
    # except fiona.errors.DriverError as e:
    #     print(f"Error: Could not open the file. Please check the path and ensure it's a valid GeoPackage file.\nDetails: {e}")


if __name__ == "__main__":
    calc_energy_per_site(config)