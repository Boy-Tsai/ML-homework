import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=40, centers=centers, cluster_std=0.6,random_state=0)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

centers1 = kmeans.cluster_centers_
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap='viridis')
plt.scatter(centers1[:, 0], centers1[:, 1], c='black', s=25, alpha=0.5,marker="x");
plt.title("K-means")

plt.subplot(122)
colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("MeanShift clusters: %d" % n_clusters_)
plt.show()

