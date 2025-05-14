import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

train_image_names = []
train_images = []
train_labels = []

test_image_names = []
test_images = []
test_labels = []

# tempArray = np.zeros((7735, 256, 256, 3), dtype=int, order='C')

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Load the image names from the file in the correct order, as well as the labels in the correct and thus corresponding order
with open("../train_images.csv", "r") as train_image_names_file:
  # Read and print the entire file line by line
  line = train_image_names_file.readline()
  while line != '':  # The EOF char is an empty string
    line = line[:len(line) - 1]
    train_image_names.append(line)
    line = train_image_names_file.readline()

with open("../train_labels.csv", "r") as train_labels_file:
  # Read and print the entire file line by line
  line = train_labels_file.readline()
  while line != '':  # The EOF char is an empty string
    label_arr = [int(line[:len(line) - 1])]
    train_labels.append(label_arr)
    line = train_labels_file.readline()

train_labels = np.array(train_labels)

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for image_name in train_image_names:
  train_images.append(np.array(Image.open(f"KerasTunerTrainImages/{image_name}")))
train_images = np.asarray(train_images)

# Load the image names from the file in the correct order, as well as the labels in the correct and thus corresponding order
with open("../test_images.csv", "r") as test_image_names_file:
  # Read and print the entire file line by line
  line = test_image_names_file.readline()
  while line != '':  # The EOF char is an empty string
    line = line[:len(line) - 1]
    test_image_names.append(line)
    line = test_image_names_file.readline()

with open("../test_labels.csv", "r") as test_labels_file:
  # Read and print the entire file line by line
  line = test_labels_file.readline()
  while line != '':  # The EOF char is an empty string
    label_arr = [int(line[:len(line) - 1])]
    test_labels.append(label_arr)
    line = test_labels_file.readline()

train_labels = np.array(train_labels)

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for image_name in test_image_names:
  test_images.append(np.array(Image.open(f"KerasTunerTestImages/{image_name}")))
test_images = np.asarray(test_images)

# Put the train images into the other tempArray to make it a real numpy array
# i = 0
# for image in train_images:
#   tempArray[i] = image
#   i += 1

x_train = train_images
y_train = train_labels
x_test = test_images
y_test = test_labels

from tensorflow import keras
from tensorflow.keras.layers import (
  Conv2D,
  Dense,
  Dropout,
  Flatten,
  MaxPooling2D
)

from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
import os

# Pre-processing
x_train = x_train.astype('int') / 255.
x_test = x_test.astype('int') / 255.

INPUT_SHAPE = (256, 256, 3)  # should be 415, 415, 3?
NUM_CLASSES = 6

class CNNHyperModel(HyperModel):
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes

  def build(self, hp):
    model = keras.Sequential()
    model.add(
      Conv2D(
        filters=hp.Choice(
          'num_filters1',
          values=[16, 32, 64, 128, 256],
          default=16
        ),
        kernel_size=hp.Choice(
          'kernel1',
          values=[3, 4, 5, 6],
          default=3
        ),
        activation='relu',
        input_shape=self.input_shape
      )
    )
    model.add(
      MaxPooling2D(pool_size=hp.Choice(
        'pooling1',
        values=[2, 3, 4, 5, 6],
        default=2
      ),
      strides=hp.Choice(
        'strides1',
        values=[1, 2, 3, 4, 5],
        default=1
      ),
      padding=hp.Choice(
        'padding1',
        values=['valid', 'same'],
        default='same'
      ))
    )
    model.add(
      Dropout(rate=hp.Float(
        'dropout1',
        min_value=0.0,
        max_value=0.7,
        default=0.2,
        step=0.025
      ))
    )
    model.add(
      Conv2D(
        filters=hp.Choice(
          'num_filters2',
          values=[16, 32, 64, 128, 256],
          default=64
        ),
        kernel_size=hp.Choice(
          'kernel2',
          values=[3, 4, 5, 6],
          default=3
        ),
        activation='relu',
        input_shape=self.input_shape
      )
    )
    model.add(
      MaxPooling2D(pool_size=hp.Choice(
        'pooling2',
        values=[2, 3, 4, 5, 6],
        default=2
      ),
      strides=hp.Choice(
        'strides2',
        values=[1, 2, 3, 4, 5],
        default=1
      ),
      padding=hp.Choice(
        'padding2',
        values=['valid', 'same'],
        default='same'
      ))
    )
    model.add(
      Conv2D(
        filters=hp.Choice(
          'num_filters3',
          values=[16, 32, 64, 128, 256],
          default=128
        ),
        kernel_size=hp.Choice(
          'kernel3',
          values=[3, 4, 5, 6],
          default=3
        ),
        activation='relu',
        input_shape=self.input_shape
      )
    )
    model.add(
      MaxPooling2D(pool_size=hp.Choice(
        'pooling3',
        values=[2, 3, 4, 5, 6],
        default=2
      ),
      strides=hp.Choice(
        'strides3',
        values=[1, 2, 3, 4, 5],
        default=1
      ),
      padding=hp.Choice(
        'padding3',
        values=['valid', 'same'],
        default='same'
      ))
    )
    model.add(
      Dropout(rate=hp.Float(
        'dropout2',
        min_value=0.0,
        max_value=0.7,
        default=0.2,
        step=0.025
      ))
    )
    model.add(Flatten())
    model.add(
      Dense(
        units=hp.Int(
          'units',
          min_value=8,
          max_value=512,
          step=8,
          default=512
        ),
        activation=hp.Choice(
          'dense_activation',
          values=['relu', 'tanh', 'sigmoid'],
          default='relu'
        )
      )
    )
    model.add(Dense(self.num_classes, activation='softmax'))

    model.compile(
      optimizer=keras.optimizers.Adam(
        hp.Float(
          'learning_rate',
          min_value=1e-4,
          max_value=1e-2,
          sampling='LOG',
          default=1e-3
        )
      ),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy', 'val_accuracy']
    )

    return model

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

HYPERBAND_MAX_EPOCHS = 150
MAX_TRIALS = 250
EXECUTION_PER_TRIAL = 5
SEED = 3
N_EPOCH_SEARCH = 40

tuner = Hyperband(
  hypermodel,
  max_epochs=HYPERBAND_MAX_EPOCHS,
  objective='val_accuracy',
  seed=SEED,
  executions_per_trial=EXECUTION_PER_TRIAL,
  directory='hyperband',
  project_name='BuildingIdentifier'
)

# tuner.search_space_summary()

tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)

summary = tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]  # fails here for some reason
loss, accuracy = best_model.evaluate(x_test, y_test)

f = open("../results.txt", "a")
f.write(f"{summary}\nLoss: {loss}\nAccuracy: {accuracy}")
f.close()
