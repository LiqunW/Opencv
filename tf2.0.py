import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.dataset.fashion_mnist
(train_img,train_labels),(test_img,test_labels) = fashion_mnist.load_data()

