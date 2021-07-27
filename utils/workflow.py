# This file demostrates the basic usage of the files in utils.

from utils import load_movebank_data, run_algorithm, get_nearby_settlements, get_top_n_places
from plotting import plot_range
import os
import pickle
from termcolor import cprint
import colorama
colorama.init()
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

##### 1.  process with geopandas and shapely
cprint("1.  process with geopandas and shapely", "cyan")

data, reference = load_movebank_data(
    "C:/Users/grego/Anti-Poaching Research/data/Movebank", 
    "African elephants in Etosha National Park (data from Tsalyuk et al. 2018)")


##### 2.  get temps with fuzzy matching and cluster with DBSCAN
cprint("\n2.  get temps with fuzzy matching and cluster with DBSCAN", "cyan")

centroids, clusters, percents_found = run_algorithm(data,
                                                    clustering_method="DBSCAN",
                                                    verbose=False,
                                                    r_wo=0.06, r_heat=0.2,
                                                    mp_wo=45, mp_heat=25,
                                                    )


# # ##### optionally save centroids to file
# filename = 'kruger_centroids.pkl'
# fp = os.path.join('../data/', filename)
# with open(fp, 'wb') as output:
#     pickle.dump(centroids, output)

# ##### optionally read in pre-calculated centroids
# filename = 'kruger_centroids.pkl'
# fp = os.path.join('../data/', filename)
# with open(fp, 'rb') as infile:
#     centroids = pickle.load(infile)


##### 3. Query nearby settlements with Overpass
cprint("\n3. Query nearby settlements with Overpass", "cyan")
places = get_nearby_settlements(centroids, radius=2)


##### 4. Use KMeans to get N places
cprint("\n4. Use KMeans to get N places", "cyan")
top_10 = get_top_n_places(centroids, places, n=10)

top_10 = top_10.rename(columns={"place": "type"})
print(top_10[["geometry", "name", "type", "n_centroids_in_settlement_cluster"]])


cprint("*** Ta-da! ***", "green")





