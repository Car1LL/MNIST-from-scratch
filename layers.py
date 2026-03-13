import numpy as np

class Dense:
    def __init__(self, in_features, out_features):

        # Initialize random weights
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)

        # Initialize bias
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.T @ dZ / self.X.shape[0]
        self.db = np.sum(dZ, axis=0, keepdims=True) / self.X.shape[0]
        return dZ @ self.W.T

        