
# Table of Contents

*  [Mean-Shift Clustering Example](#org9bb2859)
    -  [Imports](#org35d9363)
    -  [Generate sample data](#org075f68e)
    -  [Compute clustering with MeanShift](#orgd8a0647)
    -  [Plot result](#orge1c6a6c)
*  [MeanShift Algo Explainer](#org3d7144c)



<a id="org9bb2859"></a>

# Mean-Shift Clustering Example


<a id="org35d9363"></a>

## Imports

```python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
```

<a id="org075f68e"></a>

## Generate sample data

```python
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
```

<a id="orgd8a0647"></a>

## Compute clustering with MeanShift

```python
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters: %d" % n_clusters_)
```

<a id="orge1c6a6c"></a>

## Plot result

```python
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf

colors = ["#dede00", "#377eb8", "#f781bf"]
markers = ["x", "o", "^"]

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        markers[k],
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )
plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()
```

<a id="org3d7144c"></a>

# MeanShift Algo Explainer

Mean shift clustering is a centroid-based algorithm which aims to discover &ldquo;blobs&rdquo; in a smooth density of samples. This works by updating candidates for centroids to be the mean of the points within a given region. Further, seeding is performed by utilizing a binning technique for scalability.

In more detail, The position of the centroid candidates is iteratively adjusted using a techniqiue called hill climbing, which finds local maxima of the estimated probability density.

