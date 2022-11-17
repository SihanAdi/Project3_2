import numpy as np
import matplotlib.pyplot as plt

def pca(features, feature_number):
    features = np.asarray(features)
    mean = np.mean(features, axis=0)
    features_mean = features - mean

    covariance = np.cov(features_mean, rowvar=0)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    index = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, index]
    eigenvectors = eigenvectors[:, :feature_number]

    PcaDate = np.dot(eigenvectors.T, features_mean.T).T

    return PcaDate

