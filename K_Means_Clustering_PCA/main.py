import random
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from K_Means_Clustering_PCA.KMeans import kMean
from K_Means_Clustering_PCA.PCA import pca
from K_Means_Clustering_PCA.findK import select_K
from K_Means_Clustering_PCA.visiable import show
from file import readFile

data = readFile.read_file("lib/train_K-Means_Clustering.csv")
data = data.drop(["rn"], axis=1)
# print(data)
features = data.drop("activity", axis=1)
label = data["activity"]
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)

# select_K(features)
start = time.time()
features = pca(features, 2)

k = 2
kmeans = kMean(k)
kmeans.fit(features)
result = kmeans.predict(features)
end = time.time() - start
print('the actual“wall-clock” time', end)
d0 = {}
d1 = {}
for i in range(len(result)):
    if result[i] == 0:
        if label[i] in d0:
            d0[label[i]] += 1
        else:
            d0[label[i]] = 1
    else:
        if label[i] in d1:
            d1[label[i]] += 1
        else:
            d1[label[i]] = 1
print('first cluster', d0)
print('second cluster', d1)

show(features, k, kmeans.centroids, kmeans.cluster)


# y_test = np.asarray(y_test)
#
# k = 2
# kmeans = kMean(k)
# kmeans.fit(x_train)
# result = kmeans.predict(x_test)
# for i in range(len(y_test)):
#     if (y_test[i] == 'STANDING' or y_test[i] == 'SITTING' or y_test[i] == 'LAYING'):
#         y_test[i] = 1
#     else:
#         y_test[i] = 0
# y_test = np.asarray(y_test.astype(int))
# result = np.asarray(result.astype(int))
#
#
# print('accuracy_score', accuracy_score(y_test, result))








