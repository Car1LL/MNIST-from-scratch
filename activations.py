import numpy as np

class ReLU:
    
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dX):
        dX_out = dX * (self.X > 0)
        return dX_out
    
def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)