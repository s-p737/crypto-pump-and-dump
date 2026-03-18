import numpy as np
from collections import Counter


class DecisionTreeNode:
    """
    A single node in a decision tree.
    Stores either a split condition (feature + threshold)
    or a leaf prediction (the majority class).
    """
    def __init__(self):
        self.feature_idx  = None   # which feature to split on
        self.threshold    = None   # split threshold value
        self.left         = None   # left subtree (feature <= threshold)
        self.right        = None   # right subtree (feature > threshold)
        self.is_leaf      = False
        self.prediction   = None   # only set if is_leaf


class DecisionTree:
    """
    A single decision tree using Gini impurity to find splits.

    At each node we:
      1. Try a random subset of features (size = sqrt(n_features))
      2. For each feature, try splitting at the midpoint between
         each pair of adjacent values
      3. Pick the split that most reduces Gini impurity
      4. Recurse until max_depth or min_samples_split is reached
    """

    def __init__(self, max_depth=10, min_samples_split=5, n_features=None):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.n_features        = n_features   # number of features to consider per split
        self.root              = None

    def fit(self, X, y):
        # if n_features not specified, use sqrt(total features) — standard RF setting
        if self.n_features is None:
            self.n_features = max(1, int(np.sqrt(X.shape[1])))
        self.root = self._build(X, y, depth=0)

    def _gini(self, y):
        """
        Gini impurity = 1 - sum(p_i^2)
        Measures how mixed the classes are at a node.
        A pure node (all one class) has Gini = 0.
        """
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs  = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """
        Tries a random subset of features and finds the split
        that minimizes weighted Gini impurity across both children.
        """
        best_gain      = -1
        best_feature   = None
        best_threshold = None
        parent_gini    = self._gini(y)
        n_samples      = len(y)

        # randomly sample a subset of feature indices
        feature_indices = np.random.choice(
            X.shape[1], size=self.n_features, replace=False
        )

        for feat in feature_indices:
            values     = X[:, feat]
            thresholds = np.unique(values)

            # try midpoint between each pair of adjacent values
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2

                left_mask  = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # weighted gini after split
                weighted_gini = (
                    (len(y_left)  / n_samples) * self._gini(y_left) +
                    (len(y_right) / n_samples) * self._gini(y_right)
                )

                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feat
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build(self, X, y, depth):
        node = DecisionTreeNode()

        # stopping conditions: max depth, too few samples, or pure node
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            node.is_leaf    = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        feat, threshold = self._best_split(X, y)

        # if no valid split found, make a leaf
        if feat is None:
            node.is_leaf    = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        node.feature_idx = feat
        node.threshold   = threshold

        left_mask  = X[:, feat] <= threshold
        right_mask = ~left_mask

        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_one(self, x, node):
        """Traverses the tree for a single example."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X, n_classes=2):
        """Returns class probabilities based on leaf majority vote."""
        preds = self.predict(X)
        proba = np.zeros((len(X), n_classes))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba


class RandomForestScratch:
    """
    Random Forest: an ensemble of decision trees each trained on a
    bootstrap sample of the data.

    Prediction is by majority vote across all trees:
        F(x) = majority{ f_1(x), f_2(x), ..., f_T(x) }

    Bootstrap sampling means each tree sees a random sample (with
    replacement) of the training data, which decorrelates the trees
    and reduces variance compared to a single decision tree.

    To handle class imbalance, we use class_weight='balanced' logic:
    minority class examples are oversampled in each bootstrap sample.
    """

    def __init__(self, n_estimators=100, max_depth=10,
                 min_samples_split=5, random_state=42):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.random_state      = random_state
        self.trees             = []

    def _bootstrap_sample(self, X, y):
        """
        Draws a bootstrap sample (random sample with replacement).
        Oversamples the minority class to handle imbalance.
        """
        n = len(y)

        # oversample minority class to balance
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        # sample equal numbers from each class
        n_sample = min(len(pos_idx) * 2, n)
        pos_sample = np.random.choice(pos_idx, size=n_sample // 2, replace=True)
        neg_sample = np.random.choice(neg_idx, size=n_sample // 2, replace=True)

        idx = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(idx)

        return X[idx], y[idx]

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for i in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(f"  Built {i + 1}/{self.n_estimators} trees...")

        return self

    def predict(self, X):
        """Majority vote across all trees."""
        # shape: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote for each sample
        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def predict_proba(self, X):
        """
        Probability estimate = fraction of trees that voted for each class.
        This is what ROC-AUC uses.
        """
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        proba     = np.zeros((n_samples, 2))

        for i in range(n_samples):
            votes        = all_preds[:, i]
            proba[i, 1]  = np.mean(votes == 1)   # fraction voting pump
            proba[i, 0]  = 1 - proba[i, 1]

        return proba