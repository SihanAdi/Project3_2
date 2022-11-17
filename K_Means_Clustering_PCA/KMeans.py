import math
import random
import numpy as np


class kMean:
    def __init__(self, k):
        self.distances = None
        self.cluster = None
        self.k = k
        self.centroids = None

    def fit(self, features):

        features = np.asarray(features)
        features_number, col_num = features.shape
        self.centroids = np.zeros((self.k, col_num))
        # randomly select k centroids
        for i in range(self.k):
            index = int(np.random.uniform(0, features_number))
            self.centroids[i, :] = features[index, :]
        self.centroids = np.asarray(self.centroids)

        # the error between the sample and the cluster to which it belongs
        self.distances = np.zeros(features_number)
        # cluster to which the sample belongs
        self.cluster = np.zeros(features_number)
        succeed = False
        count = 0
        error = []
        while not succeed:
            succeed = True
            for i in range(features_number):
                min_distance = float("inf")
                j = 0
                for centroid in self.centroids:
                    arr = np.power((features[i] - centroid), 2)
                    tmp = np.sum(arr)
                    distance = np.sqrt(tmp)

                    if distance < min_distance:
                        self.distances[i] = distance
                        if self.cluster[i] != j:
                            self.cluster[i] = j
                            succeed = False
                        min_distance = distance
                    j += 1

            for i in range(self.k):
                cluster_index = np.where(self.cluster == i)
                points = features[cluster_index]
                self.centroids[i] = np.mean(points, axis=0)

            error.append(np.sum(self.distances))
            if count > 0:
                if math.isclose(error[count], error[count - 1], abs_tol=(0.0000001 * error[count - 1])):
                    break

            count += 1
            print(count)

        cur_inertia = 0
        for i in range(len(self.distances)):
            cur_inertia += np.sum((self.distances[i]) ** 2)

        print('centroids: ', self.centroids)
        print('distances: ', self.distances)
        print('cluster: ', self.cluster)
        print('inertia: ', cur_inertia)

    def predict(self, features):
        features = np.asarray(features)
        features_number = features.shape[0]
        result = np.zeros(features_number)

        for i in range(features_number):
            min_distance = float("inf")
            j = 0
            for centroid in self.centroids:
                arr = np.power((features[i] - centroid), 2)
                tmp = np.sum(arr)
                distance = np.sqrt(tmp)

                if distance < min_distance:
                    if result[i] != j:
                        result[i] = j
                    min_distance = distance
                j += 1

        return result








