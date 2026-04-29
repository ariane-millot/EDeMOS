# EDeMOS (Electricity Demand Mapping from Open-Source data)
<pre>
</pre>

## Workflow steps

### 1. Create a new environment in Conda

The file [edemos_env.yml](edemos_env.yml) will allow you to create a new environment in Conda. Running EDeMOS from this environment should avoid any issues with libraries like gdal, h3, h3pandas, and others.  To create the environment, run:
```
conda env create -f edemos_env.yml
```
You only need to do this the very first time. Thereafter you just need to run:
```
conda activate edemos_env
```
before running EDeMOS.

### 2. Choose country
Set the variable ACTIVE_COUNTRY in [config.py](config.py).

### 3. Download data sets 
As required for the specific country (see below). Files need to be named and put in folders to match the paths given in the [config.py](config.py) and config_{Country}.py

### 4. Run data conversion scripts
- [read_DHS_hh_to_df.py](Buildings/HouseholdEnergyUse/read_DHS_hh_to_df.py)
- [read_DHS_services_to_df.py](Buildings/HouseholdEnergyUse/read_DHS_services_to_df.py) 
(select appropriate labels in config file)
- [USGS_extract_data.py] (Industry/USGS_extract_data.py)
NB: For Zambia, adjust in the AFR_Mineral_Facilities.csv file the value for DsgAttr07/Nchanga Copper Smelter/Cobalt to 3000 as indicated in MemoOther

### 5. Adjust census data
In the census data file, the following colummns should be available: HH urban, rural, total, size of HH (urban and rural) and share women

### 6. Set resolution
This is set in config_{Country}.py. HEX_SIZE=5 for testing, 6 for meaningful results, 7 for best results

### 7. Run [GeoDem.ipynb](GeoDem.ipynb)
To run .ipynb file, you can use jupyter lab, to install it, run:
```
pip install jupyterlab
``` 
In your anaconda prompt, navigate to your working folder. Activate the environment by running:
```
conda activate edemos_env
```
Then run 
```
jupyter lab
```
You can then access the different scripts and run [GeoDem.ipynb](GeoDem.ipynb)

## Datasets to download

1. GADM map [^1] in Data/admin folder
2. Energy balance (UN stats) [^2] in Data/EnergyBalance folder
3. Building footprints [^3] in Buildings/Data/WorldPop folder
4. High-Resolution Electricity Access. set_lightscore_sy_xxxx.tif: Predicted likelihood that a settlement is electrified (0 to 1) [^4]. in Buildings/Data/Lighting folder
5. Relative Wealth Index (RWI) [^5]. in Buildings/Data/WealthIndex folder
6. Demographic and Health Surveys (DHS) [^6]. in Buildings/Data/DHS/[Country name] folder
7. Population Census: UN https://population.un.org/wpp/ For Zambia: Census 2022. [^7]. in Buildings/Data/Census/[Country name] folder
8. Mining GIS data [^8] in Industry/Data/mines folder
9. USGS mining production levels [^9]

Optional:
10. A high-resolution gridded dataset to assess electrification in sub-Saharan Africa [^10].
11. Gridded global Gross Domestic Product and Human Development Index datasets over 1990–2015 [^11]. 

## Data setup
To run the notebooks, download the following datasets and place them in the directory structure shown below:

```
EDeMOS_Zambia/
├── Data/
│   ├── admin/               <-- [1] GADM map
│   └── EnergyBalance/       <-- [2] Energy balance (UN stats)
├── Buildings/
│   └── Data/
│       ├── WorldPop/        <-- [3] Building footprints
│       ├── Lighting/        <-- [4] High-Res Electricity Access
│       ├── WealthIndex/     <-- [5] Relative Wealth Index (RWI)
│       └── DHS/
│           └── [Country]/   <-- [6] DHS
│       └── Census/
│           └── [Country]/   <-- [7] Population Census
├── Industry/
│   └── Data/
│       └── mines/           <-- [8] Mining location [9] Mining production
└── GeoDem.ipynb
```

## Cite this work

EDeMOS Zambia [^12].

[^1]: https://gadm.org/download_country.html
[^2]: https://data.un.org/SdmxBrowser/start
[^3]: Leasure DR, Dooley CA, Bondarenko M, Tatem AJ. 2021. peanutButter: An R package to produce rapid-response gridded population estimates from building footprints, version 1.0.0. WorldPop, University of Southampton. doi: 10.5258/SOTON/WP00717. https://github.com/wpgp/peanutButter
[^4]: Brian Min, Zachary P. O'Keeffe, Babatunde Abidoye, Kwawu Mensan Gaba, Trevor Monroe, Benjamin P. Stewart, Kim Baugh, Bruno Sánchez-Andrade Nuño, “Lost in the Dark: A Survey of Energy Poverty from Space,” Joule (2024), https://doi.org/10.1016/j.joule.2024.05.001.
[^5]: Samapriya Roy, Swetnam, T., & Saah, A. (2025). samapriya/awesome-gee-community-datasets: Community Catalog (3.2.0).
Zenodo. https://doi.org/10.5281/zenodo.14757583.
[^6]: DHS, https://dhsprogram.com/data/available-datasets.cfm
[^7]: Zamstats, 2022, https://www.zamstats.gov.zm/census/
[^8]: Padilla, A.D., Otarod, D. (Contractor), Deloach-Overton, S.W., Kemna, R. (Contractor) F., Freeman, P.A., Wolfe, E. (Contractor) R., Bird, L. (Contractor) R., Gulley, A.L., Trippi, M.H., Dicken, C., Hammarstrom, J.M., Brioche, A.S., 2021. Compilation of Geospatial Data (GIS) for the Mineral Industries and Related Infrastructure of Africa. https://doi.org/10.5066/P97EQWXP
[^9]: https://www.usgs.gov/centers/national-minerals-information-center/africa-and-middle-east#zmb
[^10]: Falchetta, G., Pachauri, S., Parkinson, S. et al. A high-resolution gridded dataset to assess electrification in sub-Saharan Africa. Sci Data 6, 110 (2019). https://doi.org/10.1038/s41597-019-0122-6.
[^11]: Kummu, M., Taka, M. & Guillaume, J. Gridded global datasets for Gross Domestic Product and Human Development Index over 1990–2015. Sci Data 5, 180004 (2018). https://doi.org/10.1038/sdata.2018.4.
[^12]: Millot, A., Kerekeš, A., Korkovelos, A., Stringer M., and Hawkes A. EDeMOS_Zambia. GitHub repository. Accessed February 10, 2025. https://github.com/ariane-millot/EDeMOS_Zambia.