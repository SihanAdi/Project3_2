import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def select_K(features):
    ks = range(1, 10)
    inertia = []
    features = np.asarray(features)
    # Using sklearn's KMeans class
    for k in ks:
        km = KMeans(n_clusters=k)
        km.fit(features)
        centroids = km.cluster_centers_

        print(centroids)
        centroid_pre = km.predict(features)
        cur_inertia = 0

        for i in range(len(features)):
            cur_centroid = centroids[centroid_pre[i]]
            cur_inertia += np.sum((features[i] - cur_centroid) ** 2)
        inertia.append(cur_inertia)


    plt.figure()
    plt.style.use('bmh')
    plt.plot(ks, inertia, '-o', color='r')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.xticks(ks)
    plt.show()
