from sklearn import cluster
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.lines
from sklearn.decomposition import PCA
import seaborn
import numpy
from sklearn.metrics import silhouette_samples, silhouette_score

# Zadanie 1 (1 pkt.)
print('W pocie czoła ładuję dane...')
cols = ['Normalized {}'.format(i) for i in range(52)] + ['Product_Code']
source_dataset = pandas.read_csv("Sales_Transactions_Dataset_Weekly.csv", usecols=cols, index_col='Product_Code')

# Zadanie 2 (3 pkt.)
pca_2d = PCA(n_components=2)
markers = list(matplotlib.lines.Line2D.markers.keys())
colors = matplotlib.cm.get_cmap('Pastel1')

no_of_clusters = input('Drogi użyszkodniku, podaj liczbę clustrów [default: 5]:')
if no_of_clusters in ['', ' ']:
    no_of_clusters = 5
else:
    no_of_clusters = int(no_of_clusters)

# KMeans
sample_dataset = source_dataset.copy()
vis_dataset = pandas.DataFrame(pca_2d.fit_transform(sample_dataset))
cluster_classification = cluster.KMeans(no_of_clusters, init='random', random_state=170).fit(sample_dataset)
sample_dataset['cluster'] = cluster_classification.labels_
vis_dataset['cluster'] = cluster_classification.labels_
centroids = pandas.DataFrame(pca_2d.fit_transform(cluster_classification.cluster_centers_))
subplot1 = centroids.plot.scatter(0, 1, marker='x', s=169, linewidths=3, color='b', zorder=8, label='Centroids')
for i in range(no_of_clusters):
    v = vis_dataset.loc[vis_dataset['cluster'] == i]
    subplot1.scatter(v[0], v[1], color=colors(i), marker=markers[i], label='Cluster {}'.format(i))
subplot1.legend(loc='upper right')
subplot1.grid(True)
print(sample_dataset)

# KMeans++
sample_dataset = source_dataset.copy()
vis_dataset = pandas.DataFrame(pca_2d.fit_transform(sample_dataset))
cluster_classification = cluster.KMeans(no_of_clusters, init='k-means++', random_state=170).fit(sample_dataset)
sample_dataset['cluster'] = cluster_classification.labels_
vis_dataset['cluster'] = cluster_classification.labels_
centroids = pandas.DataFrame(pca_2d.fit_transform(cluster_classification.cluster_centers_))
subplot1 = centroids.plot.scatter(0, 1, marker='x', s=169, linewidths=3, color='b', zorder=8, label='Centroids')
for i in range(no_of_clusters):
    v = vis_dataset.loc[vis_dataset['cluster'] == i]
    subplot1.scatter(v[0], v[1], color=colors(i), marker=markers[i], label='Cluster {}'.format(i))
subplot1.legend(loc='upper right')
subplot1.grid(True)
print(sample_dataset)

# Zadanie 3
metryka_pomiaru_odleglosci = input('Drogi użyszkodniku, podaj metrykę pomiaru odległości [default: euclidean]: ')
if metryka_pomiaru_odleglosci in ['', ' ']:
    metryka_pomiaru_odleglosci = 'euclidean'
if metryka_pomiaru_odleglosci not in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
                                      'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                                      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
    print('Zrobiłeś mi przykrość, wychodzę :P')
    exit(0)

pca_1d = PCA(n_components=1)
colors = matplotlib.cm.get_cmap('Dark2')

# Agglomerative Clustering
sample_dataset = source_dataset.copy()
seaborn.clustermap(sample_dataset, method='average', metric=metryka_pomiaru_odleglosci)

# DBScan
sample_dataset = source_dataset.copy()
vis_dataset = pandas.DataFrame(pca_2d.fit_transform(sample_dataset))
dbscan = cluster.DBSCAN(metric=metryka_pomiaru_odleglosci, eps=0.12, min_samples=6).fit(vis_dataset)
sample_dataset['cluster'] = dbscan.labels_
vis_dataset['cluster'] = dbscan.labels_
subplot_dbscan = plt.subplot()
for i, j in enumerate(numpy.unique(dbscan.labels_)):
    if j == -1:
        continue
    v = vis_dataset.loc[vis_dataset['cluster'] == j]
    subplot_dbscan.scatter(v[0], v[1], color=colors(i), marker=markers[i], label='Cluster {}'.format(j))
subplot_dbscan.legend(loc='upper right')
subplot_dbscan.grid(True)
print(sample_dataset)

# Zadanie 4
# A teraz nauczę się czym jest silhouette coeficiency, jest 4:36 i  padam na twarz :P
# edit: poniższe zadanie zostało zrealizowane tylko dla algorytmów, które zakładają zadeklarowanie ilości
# docelowych klastrów
sample_dataset = source_dataset.copy()
vis_dataset = pandas.DataFrame(pca_2d.fit_transform(sample_dataset))
for n_clusters in range(2, 8):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(vis_dataset) + (n_clusters + 1) * 10])
    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(vis_dataset)

    silhouette_avg = silhouette_score(vis_dataset, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(vis_dataset, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = matplotlib.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # wizualizacja rozkladu klastrów
    colors = matplotlib.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(vis_dataset[0], vis_dataset[1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

# Koniec
plt.show()