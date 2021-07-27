# Elephants DBSCAN Research
### Gregory Glatzer

This repository accompanies the research paper **An Analysis of Elephants’ Movement Data in Sub-Saharan Africa Using Clustering**

## Contents

1. [Notebooks](#notebooks)
2. [Utils](#utils)
3. [Data](#data)
4. [Supplement - Streamlit Application](#supplement)

---

## Notebooks

This folder contains the major Jupyter notebooks used for the research. In order of creation the notebooks are:

1. Kruger EDA
2. Kruger ML
3. External temperature data methods for analyzing African Elephant movement with DBSCAN
4. Fuzzy matching

**Notebooks 1 and 2** are more or less precursors to the research paper, but are still worthwhile to see how I got to my conclusions. **Notebook 3** is the majority of the work, with the implementations of Temp-Influenced and Without Temp-influence clustering, as well as timestamp matching, outlined in the notebook. Fuzzy timestamp matching is implemented in **Notebook 4**.


## Utils

This folder contains the final form of lots of the functions used throughout the notebooks. If you want to use any of this code in another project, I would highly recommend using the code from `utils.py` found inside this folder, instead of code found in the notebooks. Some quirks and edge cases were fixed along the way, and the best version of the majority of the functions exists in this file. For example, `load_movebank_data()` exists in multiple notebooks, but the version that should be used is the one in `utils.py`. 

The utils folder contains a `workflow.py`. This file demonstrates how the functions in `utils.py` should be used
in order to recreate the steps outlined in the paper. The steps in `workflow.py` are also outlined in the a flowchart found at `supplement/flowchart.png`. It is important to note that step 4 in `workflow.py` (also present in the  flowchart) is not mentioned in the research paper. This step takes the centroids and nearby places and uses KMeans to classify each centroid to a place. Doing so allows us to programtically rank each place based on how many elephants are near the given location. This final step outputs the "locations of interest" as talked about in the paper.


## Data

In order to run these files, you will need the data. The data is publicly available at these links:

- Kruger    
    - dataset: http://dx.doi.org/10.5441/001/1.403h24q5/1
    - reference: http://dx.doi.org/10.5441/001/1.403h24q5/2

- Etosha
    - dataset http://dx.doi.org/10.5441/001/1.3nj3qj45/1
    - reference http://dx.doi.org/10.5441/001/1.3nj3qj45/2

- Wall J et al
    - dataset: http://dx.doi.org/10.5441/001/1.f321pf80/1
    - reference: http://dx.doi.org/10.5441/001/1.f321pf80/2

- Blake et al
    - dataset: https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study1818825 
    - reference: https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study1818825 


The data is loaded in by `load_movebank_data()`, provided in `utils.utils.py`. All of the necessary files can be found from the links above. The code expects the data to be in a certain file structure, as illustrated below:

```
📦Movebank
 ┣ 📂African elephants in Etosha National Park (data from Tsalyuk et al. 2018)
 ┃ ┣ 📜African elephants in Etosha National Park (data from Tsalyuk et al. 2018)-reference-data.csv
 ┃ ┗ 📜African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv
 ┣ 📂Forest Elephant Telemetry Programme
 ┃ ┣ 📜Forest Elephant Telemetry Programme-reference-data.csv
 ┃ ┗ 📜Forest Elephant Telemetry Programme.csv
 ┣ 📂ThermochronTracking Elephants Kruger 2007
 ┃ ┣ 📜ThermochronTracking Elephants Kruger 2007-reference-data.csv
 ┃ ┗ 📜ThermochronTracking Elephants Kruger 2007.csv
 ┗ 📂Elliptical Time-Density Model (Wall et al. 2014) African Elephant Dataset (Source-Save the Elephants)
   ┣ 📜Elliptical Time-Density Model (Wall et al. 2014) African Elephant Dataset (Source-Save the Elephants)-reference-data.csv
   ┗ 📜Elliptical Time-Density Model (Wall et al. 2014) African Elephant Dataset (Source-Save the Elephants).csv
```

## Supplement - Streamlit Application <a name="supplement"></a>

The discussion section of the research paper has images from a folium map embedded in a Streamlit application. That application can be found at 

https://share.streamlit.io/g1776/elephantcentroids/main/app.py 


Additional screenshots from the application, as well as some screenshots from maps of the clustering and centroids, can be found in the supplement folder in this repository.
