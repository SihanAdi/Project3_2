from collections import Counter

import numpy as np

from Random_Forest.ID3 import ID3


class RandomForest:

    def __init__(self, trees_number, min_samples_split=2, max_depth=5):
        self.trees_number = trees_number
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # trained decision trees
        self.decision_trees = []

    def fit(self, features, labels):

        if len(self.decision_trees) > 0:
            self.decision_trees = []
        n_rows, n_cols = features.shape
        num_built = 0
        # construct each tree
        while num_built < self.trees_number:
            print(num_built)
            decisionTree = ID3(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            index = np.random.choice(a=n_rows, size=n_rows, replace=True)
            feature = features.iloc[index].values
            label = labels.iloc[index].values
            decisionTree.fit(feature, label)
            self.decision_trees.append(decisionTree)
            num_built += 1

    def predict(self, test_data):
        predictions = []
        for tree in self.decision_trees:
            predictions.append(tree.predict(test_data))
        predictions = np.swapaxes(a=predictions, axis1=0, axis2=1)

        final_predictions = []
        for sub_predictions in predictions:
            final_predictions.append(Counter(sub_predictions).most_common(1)[0][0])
        return np.array(final_predictions)
