import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_estimators=500, max_depth=13, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators  # Number of trees in the forest
        self.max_depth = max_depth  # Maximum depth of each tree
        self.max_features = max_features  # Maximum number of features to consider for splitting
        self.random_state = random_state  # Random seed for reproducibility
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = [self._build_tree(X, y) for _ in range(self.n_estimators)]

    def _build_tree(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            raise ValueError("Invalid value for max_features")

        indices = np.random.choice(n_features, max_features, replace=False)
        X_subset = X[:, indices]
        tree = self._create_tree(X_subset, y, depth=0)
        return tree

    def _create_tree(self, X, y, depth):
        # Stopping criteria
        if len(set(y)) == 1 or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]

        # Find the best split
        best_gain = 0
        best_feature, best_threshold = None, None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                y_left = y[X[:, feature_index] <= threshold]
                y_right = y[X[:, feature_index] > threshold]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_gain == 0:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left_tree = self._create_tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self._create_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feature, best_threshold, left_tree, right_tree)

    def _information_gain(self, parent, left_child, right_child):
        # Calculate information gain using entropy
        p = len(left_child) / len(parent)
        entropy_parent = self._entropy(parent)
        entropy_children = p * self._entropy(left_child) + (1 - p) * self._entropy(right_child)
        return entropy_parent - entropy_children

    def _entropy(self, y):
        # Calculate entropy
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        predictions = [self._predict_tree(x, self.trees) for x in X]
        return np.array(predictions)
    
    def _predict_tree(self, x, trees):
        predictions = [self._traverse_tree(x, tree) for tree in trees]
        return Counter(predictions).most_common(1)[0][0]

    def predict_proba(self, X):
        proba = []
        for x in X:
            predictions = [self._traverse_tree(x, tree) for tree in self.trees]
            count = Counter(predictions)
            total = sum(count.values())
            proba.append({cls: count[cls] / total for cls in count})
        return np.array(proba)

    def _traverse_tree(self, x, node):
        if isinstance(node, int):
            return node  # Leaf node, return the predicted class

        # Unpack the node values
        try:
            feature_index, threshold, left_tree, right_tree = node
        except ValueError:
            # If the node structure is different, handle it accordingly
            return node  # Return the node as it is

        # Check if the feature_index is valid for accessing elements in x
        if isinstance(feature_index, int) and 0 <= feature_index < len(x):
            if x[feature_index] <= threshold:
                return self._traverse_tree(x, left_tree)
            else:
                return self._traverse_tree(x, right_tree)
        else:
            return node  # Return the node as it is