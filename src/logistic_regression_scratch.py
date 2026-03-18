import numpy as np


# Models pump probability using the sigmoid function:
#   P(y=1|x) = 1 / (1 + e^(-(w^T x + b)))
#
# Trained by minimizing binary cross-entropy loss:
#   L(w, b) = -1/n * sum[ y*log(p) + (1-y)*log(1-p) ]
#
# Gradient updates:
#   w := w - alpha * (1/n) * X^T (p - y)
#   b := b - alpha * (1/n) * sum(p - y)
#
# To handle class imbalance, we weight the positive class (pumps)
# higher in the loss function using class_weight='balanced' logic.
class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.1, n_iterations=1000,
                 random_state=42, class_weight='balanced'):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.random_state  = random_state
        self.class_weight  = class_weight
        self.weights       = None
        self.bias          = None
        self.scaler_mean   = None
        self.scaler_std    = None

    # sigmoid maps any real number to (0, 1) — gives us a valid probability
    # clip prevents overflow in exp for very large or very negative z values
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # upweights minority class (pumps) proportionally so the model doesn't
    # just predict normal for everything — formula: n / (n_classes * count)
    def _compute_class_weights(self, y):
        n_samples  = len(y)
        n_classes  = 2
        n_pos      = np.sum(y == 1)
        n_neg      = np.sum(y == 0)

        weight_pos = n_samples / (n_classes * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_samples / (n_classes * n_neg) if n_neg > 0 else 1.0

        sample_weights = np.where(y == 1, weight_pos, weight_neg)
        return sample_weights

    # standardizes features to zero mean and unit variance
    # logistic regression is sensitive to feature scale — without this,
    # large features like vol_spike_ratio would dominate the gradient updates
    # fit=True on training data, False on test (reuse training mean/std)
    def _normalize(self, X, fit=False):
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std  = np.std(X, axis=0) + 1e-8   # epsilon avoids divide-by-zero
        return (X - self.scaler_mean) / self.scaler_std

    # trains the model using gradient descent
    # X: (n_samples, n_features), y: (n_samples,) binary labels
    def fit(self, X, y):
        np.random.seed(self.random_state)

        X = self._normalize(X, fit=True)
        n_samples, n_features = X.shape

        # initialize weights to zero — gradient descent will move them from here
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        # compute per-sample weights to correct for class imbalance
        if self.class_weight == 'balanced':
            sample_weights = self._compute_class_weights(y)
        else:
            sample_weights = np.ones(n_samples)

        for i in range(self.n_iterations):
            # forward pass: linear combination through sigmoid gives probabilities
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)

            # error = how far off our predicted probability is from the true label
            # multiply by sample_weights so pump errors are penalized more heavily
            error          = p - y
            weighted_error = error * sample_weights

            grad_w = (1 / n_samples) * (X.T @ weighted_error)
            grad_b = (1 / n_samples) * np.sum(weighted_error)

            # gradient descent step — move weights opposite to the gradient
            self.weights -= self.learning_rate * grad_w
            self.bias    -= self.learning_rate * grad_b

            # log loss every 100 iterations to confirm it's converging
            # clip p to avoid log(0) which would be -inf
            if (i + 1) % 100 == 0:
                p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
                loss = -np.mean(
                    sample_weights * (
                        y * np.log(p_clipped) +
                        (1 - y) * np.log(1 - p_clipped)
                    )
                )
                print(f"  Iteration {i+1}/{self.n_iterations} | Loss: {loss:.4f}")

        return self

    # returns probabilities for both classes — column 0 = P(normal), column 1 = P(pump)
    # stacking both columns makes this compatible with sklearn's predict_proba interface
    def predict_proba(self, X):
        X      = self._normalize(X)
        z      = X @ self.weights + self.bias
        p_pump = self._sigmoid(z)
        return np.column_stack([1 - p_pump, p_pump])

    # predicts hard labels using a probability threshold
    # default 0.5, but lowering it increases recall at the cost of more false positives
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)