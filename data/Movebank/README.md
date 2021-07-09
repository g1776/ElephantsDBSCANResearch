## Data Disclaimer

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


The data is loaded in by `load_movebank_data()`, provided in `utils.utils.py`. All of the necessary files can be found from the links above.
