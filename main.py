from layers import Dense
from activations import ReLU, softmax
from loss import Loss
from optimization import SGD
import numpy as np
from data import load_mnist_images, load_mnist_labels
import matplotlib.pyplot as plt


batch_size = 32
EPOCHS = 10

X_train = load_mnist_images('./MNIST/raw/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('./MNIST/raw/train-labels-idx1-ubyte.gz')

X_test = load_mnist_images('./MNIST/raw/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('./MNIST/raw/t10k-labels-idx1-ubyte.gz')


# Define model architecture (amount of layers, hidden units and activation functions)

layer1 = Dense(in_features=784, out_features=512) # image 28x28 -> 784
activation1 = ReLU()
layer2 = Dense(in_features=512, out_features=256)
activation2 = ReLU()
layer3 = Dense(in_features=256, out_features=128)
activation3 = ReLU()
layer4 = Dense(in_features=128, out_features=10) # 0-9 digits -> 10 classes

layers = [layer1, layer2, layer3, layer4] # To use with optimizer

loss_fn = Loss()

optimizer = SGD(parameters=layers, lr=0.01)

def train_loop(X_train, y_train, batch_size, epochs=1):
    train_loss = []
    # Amount of all images
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle our data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for start in range(0, num_samples, batch_size):
            # Split the data into mini-batches
            end = start + batch_size
            X_batch = X_train[start:end] # [0:32] -> [32:64] -> ... (if batch_size=32)
            y_batch = y_train[start:end]

            # Forward pass
            Z1 = layer1.forward(X_batch)
            A1 = activation1.forward(Z1)
            Z2 = layer2.forward(A1)
            A2 = activation2.forward(Z2)
            Z3 = layer3.forward(A2)
            A3 = activation3.forward(Z3)
            logits = layer4.forward(A3)
            

            loss = loss_fn.forward(logits=logits, y_true=y_batch)
            epoch_loss += loss * X_batch.shape[0]

            # Backward
            dZ4 = loss_fn.backward()
            dA3 = layer4.backward(dZ4)
            dZ3 = activation3.backward(dA3)
            dA2 = layer3.backward(dZ3)
            dZ2 = activation2.backward(dA2)
            dA1 = layer2.backward(dZ2)
            dZ1 = activation1.backward(dA1)
            layer1.backward(dZ1)

            # Update parameters of the model (Weights & bias)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= num_samples
        train_loss.append(epoch_loss)
        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

    return train_loss

def test_evaluation(X_test, y_test, batch_size):
    num_samples = X_test.shape[0]

    test_loss = 0
    correct = 0

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        # Forward pass
        Z1 = layer1.forward(X_batch)
        A1 = activation1.forward(Z1)
        Z2 = layer2.forward(A1)
        A2 = activation2.forward(Z2)
        Z3 = layer3.forward(A2)
        A3 = activation3.forward(Z3)
        
        logits = layer4.forward(A3)

        # Loss
        loss = loss_fn.forward(logits=logits, y_true=y_batch)
        test_loss += loss * X_batch.shape[0]

        # Softmax predictions
        probs = softmax(logits)

        preds = np.argmax(probs, axis=1)

        correct += np.sum(preds==y_batch)

    test_loss /= num_samples
    accuracy = (correct / num_samples) * 100

    print(f"\n\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return test_loss, accuracy
    
              


def predict_single_image(X_test, y_test):
    
    # Choose random image
    idx = np.random.randint(0, X_test.shape[0])

    image = X_test[idx].reshape(1, -1) # (1, 784)
    true_label = y_test[idx]

    print(f"\n\nImage shape: {image.shape}")
    print(f"True label: {true_label}")

    plt.imshow(image.reshape(28, 28), cmap="grey")
    plt.axis(False)
    plt.show()

    Z1 = layer1.forward(image)
    A1 = activation1.forward(Z1)
    Z2 = layer2.forward(A1)
    A2 = activation2.forward(Z2)
    Z3 = layer3.forward(A2)
    A3 = activation3.forward(Z3)
    logits = layer4.forward(A3)

    # Softmax
    probs = softmax(logits)

    pred_label = np.argmax(probs, axis=1)[0]
    print(f"\n\nPredicted label: {pred_label}")

    if pred_label == true_label:
        print(f"Model identified digit correctly")
    else:
        print(f"Model identified digit incorrectly")

def plot_results(train_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss)
    plt.title("Training loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_loss = train_loop(X_train=X_train, y_train=y_train, batch_size=batch_size, epochs=EPOCHS)
    # plot_results(train_loss)
    # predict_single_image(X_test=X_test, y_test=y_test)
    test_evaluation(X_test=X_test, y_test=y_test, batch_size=batch_size)