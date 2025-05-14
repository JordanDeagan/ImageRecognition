import numpy as np
import tensorflow.python.data
from PIL import Image
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']="4"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
  Conv2D,
  Dense,
  Dropout,
  Flatten,
  MaxPooling2D
)
from keras_tuner.tuners import RandomSearch
from keras_tuner.tuners import Hyperband
from keras_tuner import HyperModel
import tensorflow_similarity as tfsim
import gc


print("THIS IS THE PROPER KERAS TUNER ///////////////////////////////////////////////////////////" \
"////////////////////////////////////////////////////////////////////////////////////////////////////////")
keras.backend.clear_session()  # these 2 are an attempt to fix resourceexchausted errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''

image_annotation = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ArtificialIntel/ImageAnnotation/"
train_dir = image_annotation+"KerasTunerTrain"
test_dir = image_annotation+"KerasTunerTest"

batch_size = 32
img_height = 256
img_width = 256

print(tf.__version__)

num_classes = 6

# train_img = tf.keras.utils.image_dataset_from_directory(
#   train_dir,
#   validation_split= 0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size
# )

#
# val_img = tf.keras.utils.image_dataset_from_directory(
#   train_dir,
#   validation_split= 0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size
# )
#
# test_img = tf.keras.utils.image_dataset_from_directory(
#   test_dir,
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size
# )


# class_names = train_img.class_names
# print(class_names)


# Pre-processing

AUTOTUNE = tf.data.AUTOTUNE

train_img = train_img.cache().prefetch(buffer_size=AUTOTUNE)
val_img = val_img.cache().prefetch(buffer_size=AUTOTUNE)
test_img = test_img.cache().prefetch(buffer_size=AUTOTUNE)

# print(x_test[0])

input_shape = (256, 256, 3)  # should be 415, 415, 3?



model = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    # layers.Conv2D(filters=32,
    #               kernel_size=3,
    #               activation='relu',
    #               input_shape=input_shape),
    # layers.MaxPooling2D(),
    # layers.Conv2D(filters=64,
    #               kernel_size=3,
    #               activation='relu',
    #               input_shape=input_shape),
    # layers.MaxPooling2D(),
    # layers.Conv2D(filters=128,
    #               kernel_size=3,
    #               activation='relu',
    #               input_shape=input_shape),
    # layers.MaxPooling2D(),
    # layers.Conv2D(filters=256,
    #               kernel_size=3,
    #               activation='relu',
    #               input_shape=input_shape),
    # layers.MaxPooling2D(),
    # layers.Flatten(),
    # layers.Dense(units=256,
    #              activation='relu'),
    # layers.Dense(units=256,
    #              activation='relu'),
    # layers.Dense(units=128,
    #              activation='relu'),
    # layers.Dense(units=128,
    #              activation='relu'),
    # layers.Dense(num_classes)
])
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

model.summary()

epochs = 10
history = model.fit(
  train_img,
  validation_data=val_img,
  epochs=epochs
)

loss, acc = model.evaluate(test_img)
print("Accuracy:", acc)
print("Loss:", loss)

if __name__ == '__main__':
  print("end")