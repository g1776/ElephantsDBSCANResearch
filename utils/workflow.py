from utils import load_movebank_data, run_algorithm, get_nearby_settlements, get_top_n_places
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

centroids, clusters, percents_found = run_algorithm(data, r_heat=0.2, mp_heat=25, 
                                                            r_wo=0.06, mp_wo=45,
                                                            verbose=False)


##### optionally save centroids to file
# centroids.to_csv('etosha_centroids.csv')

##### optionally read in pre-calculated centroids
# centroids = pd.read_csv('etosha_centroids.csv')


##### 3. Query nearby settlements with Overpass
cprint("\n3. Query nearby settlements with Overpass", "cyan")
places = get_nearby_settlements(centroids, radius=2)


##### 4. Use KMeans to get N places
cprint("\n4. Use KMeans to get N places", "cyan")
top_10 = get_top_n_places(centroids, places, n=10)
print(top_10)


cprint("*** Ta-da! ***", "green")





