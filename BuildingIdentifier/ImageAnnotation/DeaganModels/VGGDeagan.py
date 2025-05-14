import numpy as np
import tensorflow.python.data
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import models, Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
  Input,
  Conv2D,
  Dense,
  Flatten,
  MaxPooling2D
)
import ssl



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

train_img = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split= 0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_img = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split= 0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

test_img = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)


# class_names = train_img.class_names
# print(class_names)


# Pre-processing

AUTOTUNE = tf.data.AUTOTUNE

train_img = train_img.cache().prefetch(buffer_size=AUTOTUNE)
val_img = val_img.cache().prefetch(buffer_size=AUTOTUNE)
test_img = test_img.cache().prefetch(buffer_size=AUTOTUNE)

# print(x_test[0])

input_shape = (256, 256, 3)  # should be 415, 415, 3?
num_classes = 6

model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape, classes=6)
# model_vgg16_conv.summary()

model = Sequential()

#add vgg layer (inputLayer, block1, block2)
for layer in model_vgg16_conv.layers:
    model.add(layer)

# Freezing the layers (Oppose weights to be updated)
for layer in model.layers:
    layer.trainable = False


model.add(Flatten(name='flatten'))
model.add(Dense(2048, activation='relu', name='fc1'))
model.add(Dense(1024, activation='relu', name='fc2'))
model.add(Dense(num_classes, activation='softmax', name='predictions'))
print("\n\n")
model.summary()

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)


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