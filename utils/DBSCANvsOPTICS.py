## This file compares results from DBSCAN and OPTICS clustering methods using the provided plotting method.


from utils import load_movebank_data, run_algorithm
from plotting import plot_range
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import time

data, reference = load_movebank_data(
    "C:/Users/grego/Anti-Poaching Research/data/Movebank", 
    "African elephants in Etosha National Park (data from Tsalyuk et al. 2018)")


ID = "AG006"

data = data[data["tag-local-identifier"] == ID]


fig, axs = plt.subplots(nrows=1, ncols=2)


print("------- DBSCAN -------")
start = time.time()
centroids, clusters, percents_found = run_algorithm(data,
                                                    clustering_method="DBSCAN",
                                                    verbose=True,
                                                    r_wo=0.2, r_heat=0.2,
                                                    mp_wo=45, mp_heat=25,
                                                    )
print("TIME: ", time.time() - start)
plot_range(clusters, centroids, ax=axs[0], show=False)
axs[0].set_title("DBSCAN")



print("\n------- OPTICS -------")
start = time.time()
centroids, clusters, percents_found = run_algorithm(data,
                                                    clustering_method="OPTICS",
                                                    verbose=True,
                                                    r_wo=0.02, r_heat=0.23,
                                                    )

print("TIME: ", time.time() - start)
plot_range(clusters, centroids, ax=axs[1], show=False)
axs[1].set_title("OPTICS")

axs[0].get_legend().remove()
axs[1].get_legend().remove()

plt.show()




