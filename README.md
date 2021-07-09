# Elephants DBSCAN Research
### Gregory Glatzer

This repository accompanies the research paper **Applying DBSCAN to Elephant Movement Data**

## Contents

### Notebooks

This folder contains the major Jupyter notebooks used for the research. In order of creation the notebooks are:

1. Kruger EDA
2. Kruger ML
3. External temperature data methods for analyzing African Elephant movement with DBSCAN
4. Fuzzy matching

**Notebooks 1 and 2** are more or less precursors to the research paper, but are still worthwhile to see how I got to my conclusions. **Notebook 3** is the majority of the work, with the implementations of Temp-Influenced and Without Temp-influence clustering, as well as timestamp matching, outlined in the notebook. Fuzzy timestamp matching is implemented in **Notebook 4**.

### Utils





---

 ## Data

In order to run these files, you will need the data. The data is publicly available at these links:

- Kruger    
    - dataset: http://dx.doi.org/10.5441/001/1.403h24q5/1
    - reference: http://dx.doi.org/10.5441/001/1.403h24q5/2

- Etosha
    - dataset http://dx.doi.org/10.5441/001/1.3nj3qj45/1
    - reference http://dx.doi.org/10.5441/001/1.3nj3qj45/2

- Forest Elephants
    - dataset: http://dx.doi.org/10.5441/001/1.f321pf80/1
    - reference: http://dx.doi.org/10.5441/001/1.f321pf80/2


The code expects the data to be in a certain file structure, as illustrated below:

```
📦Movebank
 ┣ 📂African elephants in Etosha National Park (data from Tsalyuk et al. 2018)
 ┃ ┣ 📜African elephants in Etosha National Park (data from Tsalyuk et al. 2018) README.txt
 ┃ ┣ 📜African elephants in Etosha National Park (data from Tsalyuk et al. 2018)-reference-data.csv
 ┃ ┗ 📜African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv
 ┣ 📂Forest Elephant Telemetry Programme
 ┃ ┣ 📜Forest Elephant Telemetry Programme-reference-data.csv
 ┃ ┗ 📜Forest Elephant Telemetry Programme.csv
 ┗ 📂ThermochronTracking Elephants Kruger 2007
 ┃ ┣ 📜ThermochronTracking Elephants Kruger 2007 README.txt
 ┃ ┣ 📜ThermochronTracking Elephants Kruger 2007-reference-data.csv
 ┃ ┗ 📜ThermochronTracking Elephants Kruger 2007.csv
```

The data is loaded in by `load_movebank_data()`, provided in `utils.utils.py`. All of the necessary files can be found from the links above.

