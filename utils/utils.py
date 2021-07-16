import os
from collections import Counter
from termcolor import cprint
import colorama
colorama.init()

import pandas as pd
import numpy as np
import geopandas as gpd

from pyproj import CRS
from shapely.geometry import Point
from meteostat import Stations, Hourly
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


def load_movebank_data(movebank_root, study_name, epsg=3395):
    """Loads in data from the movebank folder, with the specified folder structure.

    Args:
        movebank_root (str): fp to the movebank folder.
        study_name (str): full name of the study folder to load from.
        epsg (int, optional): EPSG id of the desired projection. Defaults to 3395.

    Returns:
        Tuple: The first element is the data as a GeoDataFrame, the second is the reference data as a DataFrame.
    """


    # Default epsg:3395 (Mercator)
    
    # Define parts of filepaths
    root_folder = movebank_root
    study_data = study_name + ".csv"
    reference_data = study_name + "-reference-data.csv"
    
    
    # build filepaths
    study_fp = os.path.join(root_folder, study_name, study_data)
    reference_fp = os.path.join(root_folder, study_name, reference_data)

    
    # load data
    study_df = pd.read_csv(study_fp)
    reference_df = pd.read_csv(reference_fp)
    
    # Create shapely Points
    study_df["geometry"] = study_df.apply(lambda row: Point([row["location-long"], row["location-lat"]]), axis=1)
    
    # Create gdf and assign CRS
    study_gdf = gpd.GeoDataFrame(study_df, geometry="geometry")
    study_gdf.crs = CRS(f"epsg:{epsg}").to_wkt()
    
    # cast timestamp to dt
    study_gdf["timestamp"] = pd.to_datetime(study_gdf["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    
    
    return study_gdf, reference_df


def get_station_temps(elephant_data, num_stations=10, fuzzy=True):
    """
    Get historical temperature data for all data points from a local weather station.
    
    Parameters
    ------------
    elephant_data: (DataFrame) Contains at least the following columns: ["location-lat", "location-long", "timestamp"]
    num_stations: (int) The number of stations to search through. Default 10.
    fuzzy: (bool) Use fuzzy timestamp matching. Highly recommended. Default True
    
    Returns
    ------------
    heat_joined: (DataFrame) The original elephant_data with a new stationTemp column. None if 0 stations.
    closest_station: (DataFrame) Metadata describing the weather station that the temperature data came from (from the meteostat package. None if not stations found.
    extra: (DataFrame) Some values used throughouth the calculations that may be of interest. The values are:
    
        lat: The median latitude of elephant_data, used to find the nearest weather station
        long: The median longitude of elephant_data, used to find the nearest weather station 
        start: The earliest date in elephant_data, used to narrow down the possible stations
        end: The latest date in elephant_data, used to narrow down the possible stations
        distance: The euclidean distance between the coords of the median of elephant_data, and the weather station coords. -1 if no stations.
    
    """

    lat = elephant_data["location-lat"].median() # take median to avoid outliers
    long = elephant_data["location-long"].median() # take median to avoid outliers
    start = elephant_data.timestamp.min().to_pydatetime()
    end = elephant_data.timestamp.max().to_pydatetime()
        
    # Get nearby weather stations
    stations = Stations()
    stations_query = stations.nearby(lat, long)
    stations = stations_query.fetch(num_stations)

    # Filter to stations with data in the timeframe
    stations = stations[stations["hourly_start"].notnull()]
    possible_stations = stations[(stations["hourly_start"] <= start) & (stations["hourly_end"] >= end)]
    
    # calculate distance to study for each station
    get_distance = lambda row: Point(row.longitude, row.latitude).distance(Point(long, lat))
    possible_stations["distance"] = possible_stations.apply(get_distance, axis=1)
    

    # find closest station with data
    possible_stations.sort_values("distance", ascending = False, inplace=True) # sort by distance
    closest_station = None
    for _, station in possible_stations.iterrows():
        wmo = station.wmo
        query = Hourly(wmo, start, end, model=False) # build query
        query = query.normalize()
        query = query.interpolate() # fill in gaps in data
        station_data = query.fetch() # the actual API call
        if station_data.shape[0] > 0:
            closest_station = station_data
            closest_distance = station.distance
            print(f"Using station data from Station(wmo = {station.wmo}) at distance {round(closest_distance, 3)}")
            break
    
    if closest_station is None:
        print("No stations found")
        heat_joined = None
        closest_distance = -1
        closest_station = None
    else:
        wmo_heat = closest_station[["temp"]]
        tol = elephant_data.timestamp.diff().median() / 2
        
        # pd.merge_asof requires sorting beforehand
        elephant_data.sort_values("timestamp", inplace=True)
        wmo_heat.sort_index(inplace=True)
        
        if fuzzy:
            print(f"Fuzzy tolerance: {tol}")
            heat_joined = pd.merge_asof(right=wmo_heat, left=elephant_data, right_index=True, left_on="timestamp", tolerance=tol, direction="forward").reset_index(drop=True)
        else:
            heat_joined = pd.merge(left=elephant_data, right=wmo_heat, left_on="timestamp", right_index=True, how="left").reset_index(drop=True)
        
        heat_joined.rename(columns={"temp": "stationTemp"}, inplace=True)
        if heat_joined[heat_joined.stationTemp.notna()].shape[0] == 0:
            print("No timestamps found")
            heat_joined = None
    
    # return some info about the calculations
    extra = {"lat": lat, "long": long, "start": start, "end": end, "distance": closest_distance}
    
    return heat_joined, closest_station, pd.DataFrame(extra, index=[0])


def perform_DBSCAN(data, radius, min_points, noise, cols):

    subset = data[cols]
    scaled = StandardScaler().fit_transform(subset)

    # perform DBSCAN 
    db = DBSCAN(eps=radius, min_samples=min_points).fit(scaled)
    
    # add cluster labels
    labels = db.labels_
    data["cluster"] = labels
    
    if not noise:
        return data[data["cluster"] != -1]

    return data
    

def get_clusters(data, cols, r = 0.2, mp = 50, noise=False):
    """
    calls perform_DBSCAN and calculates centroids
    noise: return the datapoints that are not in any clusters (aka noise)
    """

    # Apply DBSCAN
    clusters = perform_DBSCAN(data, 
                            radius=r, 
                            min_points=mp,
                            noise=noise,
                            cols=cols
                            )

    # calculate centroids
    grouped = clusters.groupby("cluster")
    centroids = grouped[cols].apply(np.mean)
    centroids.index.name = "index"
    centroids["cluster"] = centroids.index
    centroids["geometry"] = centroids.apply(lambda row: Point([row["location-long"], row["location-lat"]]), axis=1)

    
    return clusters, centroids


def with_and_without_heat(data, 
                        heat_col="stationTemp",
                        noise=True,
                        r_heat=0.2, mp_heat=50, r_wo=0.1, mp_wo=35):
    
    clusters_heat, centroids_heat, clusters_wo, centroids_wo = None, None, None, None
    
    # some data points' temp will be NaN if it couldn't be found by in station data. Drop these rows.
    data_with_temps = data[data[heat_col].notna()]
    print(f"Calculating temp-influenced clusters and centroids {data_with_temps.shape}")

    clusters_heat, centroids_heat = get_clusters(data_with_temps, 
                                        ["location-long", "location-lat", heat_col],
                                        r=r_heat, mp=mp_heat, 
                                        noise=noise
                                        )
    centroids_heat["feature space"] = "Temp-influenced"


    print(f"Calculating without-temp clusters and centroids {data.shape}")
    # use all data, regardless of missing temp to calculate exclusively coordinate-based clustering
    clusters_wo, centroids_wo = get_clusters(data, 
                                        ["location-long", "location-lat"],
                                        r=r_wo, mp=mp_wo, 
                                        noise=noise
                                        )
    centroids_wo["feature space"] = "Without temp-influence"
        
    
    return [(clusters_heat, centroids_heat), (clusters_wo, centroids_wo)]


def plot_centroids(centroids, ax, hue="cluster", s_mult=1, color_legend=True):
    
    centroids_heat = centroids[centroids["feature space"] == "Temp-influenced"]
    sns.scatterplot(data = centroids_heat, 
                        x="location-long", 
                        y="location-lat",
                        color="black",
                        marker="X",
                        style="feature space",
                        style_order=["Without temp-influence", "Temp-influenced"],
                        s=85 * s_mult,
                        legend=True,
                        ax=ax
                    )
    
    # I am aware that the style attribute won't be used since I split up the data.
    # It is solely to render the legend for the different marker shapes
    centroids_wo = centroids[centroids["feature space"] == "Without temp-influence"]
    sns.scatterplot(data = centroids_wo, 
                        x="location-long", 
                        y="location-lat",
                        hue=hue,
                        palette="Paired",
                        style="feature space",
                        style_order=["Without temp-influence", "Temp-influenced"],
                        s=75 * s_mult,
                        edgecolor='black',
                        legend=color_legend,
                        linewidth=.8,
                        ax=ax
                    )


def plot_range(clusters, centroids, ax=None, show=True):
    """
    plots clusters and centroids for ONE elephant
    """
    
    # plot clusters
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(10,10))
    
    sns.set_style("white")
    sns.despine()
    sns.scatterplot(data = clusters, 
                    x="location-long", 
                    y="location-lat",
                    hue="cluster",
                    palette="Paired",
                    s=4,
                    ax=ax
                )
    
    plot_centroids(centroids, ax, color_legend=False)
    

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Legend")
    
    if show:
        plt.show()
    else: 
        return ax


def run_algorithm(data, fuzzy=True, r_heat=0.2, mp_heat=50, r_wo=0.1, mp_wo=35):
    """
    The most comprehensive form of the DBSCAN algorithm with appended historical weather station data. This function will
    run DBSCAN on the given data, as well as calculate temperature from weather stations. 
    
    Parameters
    -----------
    
    data: (DataFrame) Contains at least the columns ["location-lat", "location-long", "timestamp", "tag-local-identifier"].
    fuzzy: (bool, optional) Toggle fuzzy matching, as described in the research paper. Default True.
    
    
    Returns
    -----------
    centroids: (DataFrame) The centroids calculated (mean of values in given cluster). This is both Temp-Influenced and Without Temp-Influence.
    clusters: (DataFrame) The clusters calculated. This is only Without Temp-Influence, as the Temp-Influenced clusters are not too useful to visualize.
    percents_found: (list) List of percents of timestamps matched for each unique tag-local-identifier (in the order of data["tag-local-identifier"].unique())
    """

    centroids = None
    clusters = None
    
    all_centroids = []
    all_clusters = []
    percents_found = []

    for id, group, in data.groupby("tag-local-identifier"):
        print(id)

        station_data, station, extra = get_station_temps(group, fuzzy=fuzzy)

        # move in if no stations were found
        if station_data is None:
            print("\n")
            continue

        # calculate percent of timestamps we found temp data for
        percent_found = station_data[station_data["stationTemp"].notna()].shape[0] / group.shape[0] * 100
        print("Timestamps found: ", str(round(percent_found, 3)) + "%") 
        percents_found.append(percent_found)

        # etosha values
        # r_heat=0.2, mp_heat=25, 
        # r_wo=0.06, mp_wo=45
        (clusters_heat, centroids_heat), (clusters_wo, centroids_wo) = with_and_without_heat(station_data,
                                                                                                r_heat=r_heat, mp_heat=mp_heat, 
                                                                                                r_wo=r_wo, mp_wo=mp_wo
                                                                                            )
        centroids = centroids_heat.append(centroids_wo)
        print(f"Temp-Influenced centroids: {centroids_heat.shape[0]}")
        print(f"Without Temp-Influenced centroids: {centroids_wo.shape[0]}")
        print("\n")

        centroids["tag-local-identifier"] = id

        all_centroids.append(centroids)
        all_clusters.append(clusters_wo)
        
    if all_centroids != []:
        centroids = pd.concat(all_centroids, ignore_index=True)
    if all_clusters != []:
        clusters = pd.concat(all_clusters, ignore_index=True)
        
    
        
    return centroids, clusters, percents_found



def get_nearby_settlements(centroids, radius=1):


    print("getting human settlements")

    center_lat = centroids["location-lat"].median()
    center_long= centroids["location-long"].median()

    overpass = Overpass()

    ## bbox to get places in 
    bbox=[center_lat-radius, center_long-radius, center_lat+radius,center_long+radius]

    query = overpassQueryBuilder(
        bbox=bbox,
        elementType='node', 
        selector='place~"city|town|village|hamlet"',
        out='body'
    )

    res = overpass.query(query, timeout=50)

    places = pd.DataFrame(res.toJSON()['elements'])
    places = places.drop('tags', axis=1).join(pd.DataFrame(places.tags.values.tolist()))
    places["geometry"] = places.apply(lambda row: Point([row["lon"], row["lat"]]), axis=1)
    places = gpd.GeoDataFrame(places, geometry="geometry")

    return places


def get_top_n_places(centroids, places, n=10):

    cprint(f"Number of places: {places.shape[0]}", "cyan")

    if places.shape[0] > centroids.shape[0]:
        num_clusters = int(round(centroids.shape[0] * .75, 0))
        cprint(f"WARNING: Sampling down to {num_clusters} places, as there are more places than elephant centroids", "red")
        places = places.sample(n=num_clusters, random_state=42, replace=False)

    c_points = np.array(centroids.geometry.apply(lambda p: [p.x, p.y]).tolist())

    p_points = np.array(places.geometry.apply(lambda p: [p.x, p.y]).tolist())
    

    # use KMeans to identify how many centroids are near each settlement
    kmeans = KMeans(n_clusters=places.shape[0], init=p_points, max_iter=1)
    kmeans.fit(c_points)

    # Find Top 10 settlements based on cluster size
    counted = Counter(kmeans.labels_)
    n_counted = counted.most_common(n)
    idxs = [k for k, v in n_counted]
    top_n_places = places.iloc[idxs]
    top_n_places["n_centroids_in_settlement_cluster"] = [v for k, v in n_counted]

    
    # subset columns
    cols = ["geometry", "name", "place", "old_name", "alt_name", "n_centroids_in_settlement_cluster"]
    if "description" in top_n_places.columns:
            cols.insert(0, "description")
    try:
        top_n_places = top_n_places[cols]
    except:
        print("INVALID columns. Valid columns are:", top_n_places.columns)

    return top_n_places

                            