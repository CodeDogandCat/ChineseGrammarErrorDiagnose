# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "4,5,6,7"
set_session(tf.Session(config=config))

model = load_model('keras_test.h5.')
num_samples = 10
height = 224
width = 224
num_classes = 1000
x = np.random.random((num_samples, height, width, 3))
print("X :")
print(x)
classes = model.predict(x, batch_size=256)
print("label :")
print(classes)

