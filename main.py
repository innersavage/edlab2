from sklearn import cluster
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.lines
from sklearn.decomposition import PCA

# Zadanie 1 (1 pkt.)
print('W pocie czoła ładuję dane...')
cols = ['Normalized {}'.format(i) for i in range(52)]
source_dataset = pandas.read_csv("Sales_Transactions_Dataset_Weekly.csv", usecols=cols)

# Zadanie 2 (3 pkt.)
pca_2d = PCA(n_components=2)
markers = list(matplotlib.lines.Line2D.markers.keys())
colors = matplotlib.cm.get_cmap('Pastel1')

no_of_clusters = int(input('Drogi użyszkodniku, podaj liczbę clustrów'))

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

# Zadanie 3

plt.show()
print('Co tu się odjaniepawla?')
