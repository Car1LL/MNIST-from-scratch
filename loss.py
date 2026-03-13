import numpy as np
from activations import softmax

class Loss:
    def forward(self, logits, y_true):

        self.logits = logits
        self.y_true = y_true

        # Softmax
        self.probs = softmax(logits)

        # Calculate the loss
        batch_size = logits.shape[0]
        correct_logprobs = np.log(self.probs[np.arange(batch_size), y_true])
        loss = -np.sum(correct_logprobs) / batch_size
        return loss
    
    def backward(self):
        dZ = self.probs.copy()
        dZ[np.arange(self.logits.shape[0]), self.y_true] -= 1
        dZ /= self.logits.shape[0]
        return dZ