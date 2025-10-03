import numpy as np
from mnist import MNIST

# Initialize MNIST loader pointing to the current directory (where the .gz files should be located)
mndata = MNIST('.')
# Load training images and labels
images, labels = mndata.load_training()
# Convert to NumPy arrays and save as .npy files
np.save('mnist_data.npy', np.array(images, dtype=np.float64))  # Shape: (60000, 784)
np.save('mnist_labels.npy', np.array(labels, dtype=np.int32))
