import gzip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_mnist_images(file_path):
    with gzip.open(file_path, "rb") as f:
        f.read(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(-1, 28*28).astype(np.float32) # num_samples, 784
        data /= 255.0 # normalization into grey scale [0, 1]
        return data
    

def load_mnist_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        f.read(8)
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels
    

# def download_random_image():
#     images = load_mnist_images("./MNIST/raw/train-images-idx3-ubyte.gz")
#     img = images[0]
#     img = img.reshape(28, 28)
#     plt.imshow(img, cmap=("grey"))
#     plt.axis(False)
#     plt.show()

#     img_uint8 = (img * 255).astype(np.uint8)
#     img_pil = Image.fromarray(img_uint8)
#     img_pil.save("./random_image.jpg")

# if __name__ == "__main__":
#     download_random_image()