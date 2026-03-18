import numpy as np


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using gradient descent.

    Models the probability of a pump event using the sigmoid function:
        P(y=1|x) = 1 / (1 + e^(-(w^T x + b)))

    Trained by minimizing binary cross-entropy loss:
        L(w, b) = -1/n * sum[ y*log(p) + (1-y)*log(1-p) ]

    Gradient updates:
        w := w - alpha * (1/n) * X^T (p - y)
        b := b - alpha * (1/n) * sum(p - y)

    To handle class imbalance, we weight the positive class (pumps)
    higher in the loss function using class_weight='balanced' logic.
    """

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

    def _sigmoid(self, z):
        """Sigmoid function: maps any real number to (0, 1)."""
        # clip to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_class_weights(self, y):
        """
        Computes sample weights to balance the class distribution.
        Minority class (pumps) gets upweighted proportionally.
        n_samples / (n_classes * count_per_class)
        """
        n_samples  = len(y)
        n_classes  = 2
        n_pos      = np.sum(y == 1)
        n_neg      = np.sum(y == 0)

        weight_pos = n_samples / (n_classes * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_samples / (n_classes * n_neg) if n_neg > 0 else 1.0

        sample_weights = np.where(y == 1, weight_pos, weight_neg)
        return sample_weights

    def _normalize(self, X, fit=False):
        """
        Standardizes features to zero mean and unit variance.
        Logistic regression is sensitive to feature scale, so without
        this, large features like vol_spike_ratio would dominate.
        """
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std  = np.std(X, axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def fit(self, X, y):
        """
        Trains the model using gradient descent.
        X: (n_samples, n_features)
        y: (n_samples,) binary labels
        """
        np.random.seed(self.random_state)

        X = self._normalize(X, fit=True)
        n_samples, n_features = X.shape

        # initialize weights to zero
        self.weights = np.zeros(n_features)
        self.bias    = 0.0

        # compute sample weights for class imbalance
        if self.class_weight == 'balanced':
            sample_weights = self._compute_class_weights(y)
        else:
            sample_weights = np.ones(n_samples)

        # gradient descent loop
        for i in range(self.n_iterations):
            # forward pass: compute predictions
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)

            # compute weighted gradients
            error = p - y
            weighted_error = error * sample_weights

            grad_w = (1 / n_samples) * (X.T @ weighted_error)
            grad_b = (1 / n_samples) * np.sum(weighted_error)

            # update weights
            self.weights -= self.learning_rate * grad_w
            self.bias    -= self.learning_rate * grad_b

            # print loss every 100 iterations
            if (i + 1) % 100 == 0:
                # weighted binary cross-entropy loss
                p_clipped = np.clip(p, 1e-9, 1 - 1e-9)
                loss = -np.mean(
                    sample_weights * (
                        y * np.log(p_clipped) +
                        (1 - y) * np.log(1 - p_clipped)
                    )
                )
                print(f"  Iteration {i+1}/{self.n_iterations} | Loss: {loss:.4f}")

        return self

    def predict_proba(self, X):
        """
        Returns probability estimates for both classes.
        Column 0 = P(normal), Column 1 = P(pump)
        """
        X      = self._normalize(X)
        z      = X @ self.weights + self.bias
        p_pump = self._sigmoid(z)
        return np.column_stack([1 - p_pump, p_pump])

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels using a probability threshold.
        Default threshold = 0.5, but can be lowered to increase recall.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)