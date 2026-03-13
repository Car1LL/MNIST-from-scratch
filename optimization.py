
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for layer in self.parameters:
            if hasattr(layer, "dW"):
                layer.dW.fill(0)
            if hasattr(layer, "db"):
                layer.db.fill(0)
    
    def step(self):
        for layer in self.parameters:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
            if hasattr(layer, "b"):
                layer.b -= self.lr * layer.db