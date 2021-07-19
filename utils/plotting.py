import seaborn as sns
import matplotlib.pyplot as plt


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