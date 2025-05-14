import numpy as np
import tensorflow.python.data
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import gc
import os

print("THIS IS THE PROPER KERAS TUNER ///////////////////////////////////////////////////////////" \
"////////////////////////////////////////////////////////////////////////////////////////////////////////")
keras.backend.clear_session()  # these 2 are an attempt to fix resourceexchausted errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''

image_annotation = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/"

train_image_names = []
train_images = []
train_labels = []
train_temp = []

test_image_names = []
test_images = []
test_labels = []
test_temp = []

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
i=0
with open("../train_labels.csv", "r") as train_labels_file:
  # Read and print the entire file line by line
  line = train_labels_file.readline()
  while line != '':  # The EOF char is an empty string
    label_arr = int(line[:len(line) - 1])
    if label_arr == 3 or label_arr == 5:
      train_labels.append(label_arr)
      train_temp.append(train_image_names[i])
    line = train_labels_file.readline()
    i+=1
train_image_names = train_temp
train_labels = np.array(train_labels)

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for i in range(int(len(train_image_names) / 4)):
  train_images.append(np.array(Image.open(image_annotation+f"KerasTunerTrainImages/{train_image_names[i]}")))
train_images = np.asarray(train_images)

# i=1
# print(Image.open(image_annotation + f"KerasTunerTrainImages/{train_image_names[i]}"))

# Load the image names from the file in the correct order, as well as the labels in the correct and thus corresponding order
with open("../test_images.csv", "r") as test_image_names_file:
  # Read and print the entire file line by line
  line = test_image_names_file.readline()
  while line != '':  # The EOF char is an empty string
    line = line[:len(line) - 1]
    test_image_names.append(line)
    line = test_image_names_file.readline()

i=0

with open("../test_labels.csv", "r") as test_labels_file:
  # Read and print the entire file line by line
  line = test_labels_file.readline()
  while line != '':  # The EOF char is an empty string
    label_arr = int(line[:len(line) - 1])
    if label_arr == 3 or label_arr == 5:
      test_labels.append(label_arr)
      test_temp.append(test_image_names[i])
    line = test_labels_file.readline()
    i+=1

test_labels = np.array(test_labels)
test_image_names = test_temp

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for i in range(int(len(test_image_names) / 4)):
  test_images.append(np.array(Image.open(image_annotation+f"KerasTunerTestImages/{test_image_names[i]}")))
test_images = np.asarray(test_images)

# print("train_images: "+ str(train_images.size) + "\ntrain_labels: " + str(train_labels.size)+ "\ntrain_image_names: " + str(len(train_image_names)))
# print("test_images: "+ str(test_images.size) + "\ntest_labels: " + str(test_labels.size)+ "\ntest_image_names: " + str(len(test_image_names)))

# Put the train images into the other tempArray to make it a real numpy array
# i = 0
# for image in train_images:
#   tempArray[i] = image
#   i += 1

x_train = train_images
y_train = train_labels
x_test = test_images
y_test = test_labels
# print(x_train.size)
# print("x_train: "+ str(x_train.size) + "\ny_train: " + str(y_train.size)+ "\ntrain_image_names: " + str(len(train_image_names)))
# print("x_test: "+ str(x_test.size) + "\ny_test: " + str(y_test.size)+ "\ntest_image_names: " + str(len(test_image_names)))
# print(x_test[0])

from  tensorflow import keras
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
import os

# Pre-processing
x_train = x_train.astype('int') / 255.
x_test = x_test.astype('int') / 255.

# print(x_test[0])

INPUT_SHAPE = (256, 256, 3)  # should be 415, 415, 3?
NUM_CLASSES = 6

class CNNHyperModel(HyperModel):
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes

  def build(self, hp):
    model = keras.Sequential()
    model.add(
      Conv2D(filters=hp.Choice('num_filters1',values=[16, 32, 64, 128, 256],default=16),
        kernel_size=hp.Choice('kernel1',values=[3, 4, 5, 6],default=3),
        activation='relu',
        input_shape=self.input_shape)
    )
    model.add(MaxPooling2D())
    model.add(
      Dropout(rate=hp.Float('dropout1',min_value=0.0,max_value=0.7,default=0.2,step=0.025))
    )
    model.add(
      Conv2D(filters=hp.Choice('num_filters3',values=[16, 32, 64, 128, 256],default=128),
        kernel_size=hp.Choice('kernel3',values=[3, 4, 5, 6],default=3),
        activation='relu',
        input_shape=self.input_shape)
    )
    model.add(Flatten())
    model.add(
      Dense(units=hp.Int('units',min_value=8,max_value=512,step=8,default=512),
        activation=hp.Choice('dense_activation',values=['relu', 'tanh', 'sigmoid'],default='relu'))
    )
    model.add(
      Dense(self.num_classes, activation='softmax')
    )
    model.compile(
      optimizer=keras.optimizers.Adam(
        hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='LOG',default=1e-3)),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
    )

    return model

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

HYPERBAND_MAX_EPOCHS = 150
MAX_TRIALS = 300
EXECUTION_PER_TRIAL = 50
SEED = 3
N_EPOCH_SEARCH = 40

tuner = None

# AUTOTUNE = tf.data.AUTOTUNE
#
# x_train = x_train.cache().prefetch(buffer_size=AUTOTUNE)
# x_test = x_test.cache().prefetch(buffer_size=AUTOTUNE)

try:
  tuner = Hyperband(
    hypermodel,
    max_epochs=HYPERBAND_MAX_EPOCHS,
    objective='val_accuracy',
    seed=SEED,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='hyperband',
    project_name='BuildingIdentifier'
  )
except Exception as e:
  print("\n\n\n\n\n\n\n\n\n\n")
  print(e)
  gc.collect()
  del hypermodel
  del tuner
print("BELOW tuner\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
# tuner.search_space_summary()

# print("(Finished Tuning)")

tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)
summary = tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]  # fails here for some reason
loss, accuracy = best_model.evaluate(x_test, y_test)

# Some cleaning up
gc.collect()
del hypermodel
del tuner

f = open("../results.txt", "a")
f.write(f"{summary}\nLoss: {loss}\nAccuracy: {accuracy}")
f.close()


# if __name__ == '__main__':
#   print("end")