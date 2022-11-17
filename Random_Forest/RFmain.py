import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Random_Forest.random_Forest import RandomForest

data = pd.read_csv('/Users/adisihansun/Desktop/machine learning/Project3_2/lib/dig.csv')
# d = data
# for i in data.columns:
#     values, num = np.unique(data[i], return_counts=True)
#     if len(num) == 1 and num[0] == 799:
#         d = d.drop([i], axis=1)
#
# data = d.reset_index(drop=True)
# print(data)
# features = np.asarray(old_data.drop(["label"], axis=1))
# mean = np.mean(features, axis=0)
# features_mean = features - mean
# covariance = np.cov(features_mean, rowvar=0)
# eigenvalues, eigenvectors = np.linalg.eigh(covariance)
# index = np.argsort(-eigenvalues)
# eigenvectors = eigenvectors[:, index]
# eigenvectors = eigenvectors[:, :100]
# PcaDate = np.dot(eigenvectors.T, features_mean.T).T
# data = pd.DataFrame(PcaDate)
# data["label"] = old_data["label"]

for i in data.columns:
    data[i].values[data[i] > 0] = 1
Y = data["label"]
X = data.drop(["label"], axis=1)
print(X.shape[1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# print(X_train[0])
model = RandomForest(10)
start = time.time()
model.fit(X_train, Y_train)
pre_y = model.predict(X_test)
end = time.time() - start
print('the actual“wall-clock” time: ', end)
print("==========")
cm = metrics.confusion_matrix(Y_test, pre_y)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()
accuracy = accuracy_score(Y_test, pre_y)
print('accuracy', accuracy)



