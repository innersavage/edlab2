from sklearn import cluster
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.lines
from sklearn.decomposition import PCA
import seaborn
import numpy

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

# Koniec
plt.show()
print('Co tu się odjaniepawla?')
