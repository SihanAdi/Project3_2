import numpy as np
from collections import Counter

from Random_Forest.node import Node


class ID3:

    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def entropy(self, label):

        unique_num = np.bincount(label)
        # probabilities of each label
        probabilities = unique_num / len(label)

        probabilities = np.delete(probabilities, np.where(probabilities == 0.))
        entropy1 = np.sum(-probabilities * np.log2(probabilities))

        return entropy1

    def infoGain(self, label, feature_col, threshold):
        left = np.argwhere(feature_col <= threshold).flatten().tolist()
        right = np.argwhere(feature_col > threshold).flatten().tolist()
        if len(left) == 0 or len(right) == 0:
            return 0

        label_left = label[left]
        label_right = label[right]

        IG = self.entropy(label) - ((len(left) / len(label)) * self.entropy(label_left) + (len(right) / len(label)) * self.entropy(label_right))
        return IG

    def fit(self, feature, label):

        # build a tree
        self.root = self.buildTree(feature, label)

    def buildTree(self, feature, label, depth=0):

        feature_rows, feature_cols = feature.shape

        if depth >= self.max_depth or feature_rows < self.min_samples_split or len(np.unique(label)) == 1:
            most_label = Counter(label).most_common(1)[0][0]
            return Node(value=most_label)

        random_feature_col = np.random.choice(feature_cols, feature_cols, replace=False)
        best_IG = -1
        best_split, best_threshold = None, None
        for col in random_feature_col:
            feature_col = feature[:, col]
            for threshold in np.unique(feature_col):
                IG = self.infoGain(label, feature_col, threshold)
                if IG > best_IG:
                    best_IG = IG
                    best_split = col
                    best_threshold = threshold

        left_index = np.argwhere(feature[:, best_split] <= best_threshold).flatten().tolist()
        right_index = np.argwhere(feature[:, best_split] > best_threshold).flatten().tolist()
        left_node = self.buildTree(feature[left_index, :], label[left_index], depth + 1)
        right_node = self.buildTree(feature[right_index, :], label[right_index], depth + 1)
        return Node(best_split, best_threshold, left_node, right_node)

    def predict(self, features):
        temp = []
        for feature in features.itertuples():
            temp.append(self.findResult(feature, self.root))

        return np.array(temp)

    def findResult(self, feature, node):
        if node.value != None:
            return node.value
        if feature[node.feature] <= node.threshold:
            return self.findResult(feature, node.left_node)
        else:
            return self.findResult(feature, node.right_node)


