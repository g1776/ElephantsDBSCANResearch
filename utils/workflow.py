from utils import load_movebank_data, run_algorithm
import warnings
warnings.filterwarnings('ignore')

# process with geopandas and shapely
data, reference = load_movebank_data(
    "C:/Users/grego/Anti-Poaching Research/data/Movebank", 
    "African elephants in Etosha National Park (data from Tsalyuk et al. 2018)")

# get temps with fuzzy matching and cluster with DBSCAN
centroids, clusters, percents_found = run_algorithm(data, r_heat=0.2, mp_heat=25, 
                                                            r_wo=0.06, mp_wo=45)

# optionally save centroids to file
# centroids.to_csv('etosha_centroids.csv')

# optionally read in pre-calculated centroids
# centroids = pd.read_csv('etosha_centroids.csv')

print(centroids.head())


