
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

alpha = 0.01

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def show(images,labels, idx):
    image = images[idx]
    label = labels[idx]
    image.shape = (28,28)
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(image)
    plt.show()

show(trX,trY, 0)
