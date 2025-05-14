import numpy as np
import tensorflow.python.data
from PIL import Image
import os
import datetime
import tensorflow as tf
# import tensorflow_cloud as tfc
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    BatchNormalization
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling
)
from keras_tuner.tuners import RandomSearch
from keras_tuner.tuners import Hyperband
from keras_tuner import HyperModel
import gc

print("THIS IS THE PROPER KERAS TUNER ///////////////////////////////////////////////////////////"
      "////////////////////////////////////////////////////////////////////////////////////////////////////////")
keras.backend.clear_session()  # these 2 are an attempt to fix resourceexchausted errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''

image_annotation = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ArtificialIntel/ImageAnnotation/"
train_dir = image_annotation + "KerasTunerTrain"
test_dir = image_annotation + "KerasTunerTest"
# flower_dir = image_annotation + "flower_photos"
# cats_dogs_train = image_annotation + "cats_and_dogs_filtered/train"
# cats_dogs_val = image_annotation + "cats_and_dogs_filtered/validation"
# train_dir = image_annotation + "KerasTunerReducedTrain"
# test_dir = image_annotation + "KerasTunerReducedTest"

batch_size = 32
img_height = 256
img_width = 256
val_split = 0.1
# flower_split = 0.3
seed = 3


def resize_scale_image(image, label):
    image = tf.image.resize(image, [img_height, img_width])
    image = image/255.0
    # image = tf.image.grayscale_to_rgb(image)
    return image, label


print(tf.__version__)

train_img = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

val_img = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

test_img = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int'
)

# train_img = tf.keras.utils.image_dataset_from_directory(
#     cats_dogs_train,
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     label_mode='int'
# )
#
# val_img = tf.keras.utils.image_dataset_from_directory(
#     cats_dogs_val,
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     label_mode='int'
# )

# train_img = tf.keras.utils.image_dataset_from_directory(
#     flower_dir,
#     validation_split=flower_split,
#     subset="training",
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     label_mode='int'
# )
#
# val_img = tf.keras.utils.image_dataset_from_directory(
#     flower_dir,
#     validation_split=flower_split,
#     subset="validation",
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size,
#     label_mode='int'
# )
#
# val_batches = tf.data.experimental.cardinality(val_img)
# test_img = val_img.take(val_batches // 5)
# validation_dataset = val_img.skip(val_batches // 5)


class_names = train_img.class_names
# print(class_names)


# Pre-processing
test_resize = test_img.map(resize_scale_image)
test_raw = test_img
# preds = train_img.batch(batch_size=10).take(1)

# for a,b in test_raw.take(1):
#     print(b.numpy()[0])

#
# for a,b in preds:
#     print(str(a.shape))
# for samp in test_resize.take(10):
#     print(str(samp))

AUTOTUNE = tf.data.AUTOTUNE
train_img = train_img.cache().prefetch(buffer_size=AUTOTUNE)
val_img = val_img.cache().prefetch(buffer_size=AUTOTUNE)
test_img = test_img.cache().prefetch(buffer_size=AUTOTUNE)

# print(x_test[0])
#
input_shape = (img_height, img_width, 3)  # should be 415, 415, 3?
num_classes = len(class_names)

model = keras.Sequential([
    RandomFlip("horizontal_and_vertical", input_shape=input_shape),
    RandomRotation(0.2),
    RandomZoom(0.2),
    Rescaling(1. / 255),
    Conv2D(filters=16,
           kernel_size=3,
           # input_shape=input_shape,
           padding='same',
           activation='relu'),
    # MaxPool2D(pool_size=(2,2),
    #           strides=(2,2)),
    MaxPooling2D(),
    # Dropout(0.2),
    # BatchNormalization(),
    Conv2D(filters=32,
           kernel_size=3,
           # input_shape=input_shape,
           padding='same',
           activation='relu'),
    # MaxPool2D(pool_size=(2,2),
    #           strides=(2,2)),
    MaxPooling2D(),
    Dropout(0.2),
    # BatchNormalization(),
    Conv2D(filters=64,
           kernel_size=3,
           # input_shape=input_shape,
           padding='same',
           activation='relu'),
    # MaxPool2D(pool_size=(2,2),
    #           strides=(2,2)),
    MaxPooling2D(),
    # BatchNormalization(),
    # Conv2D(filters=256,
    #               kernel_size=3,
    #               input_shape=input_shape,
    #               activation='relu'),
    # MaxPool2D(pool_size=(2,2),
    #           strides=(2,2)),
    Dropout(0.2),
    Flatten(),
    # layers.Dense(units=256,
    #              activation='relu'),
    # layers.Dense(units=256,
    #              activation='relu'),
    Dense(units=128,
          activation='relu'),
    Dropout(0.2),
    Dense(units=128,
          activation='relu'),
    # Dropout(0.1),
    Dense(num_classes)
])

model.compile(
    optimizer='adam',
    # optimizer=keras.optimizers.Adam(learning_rate=5e-03),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

epochs = 20
history = model.fit(
    train_img,
    validation_data=val_img,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

loss, acc = model.evaluate(test_img)
print("Accuracy:", acc)
print("Loss:", loss)

# image_batch, label_batch = test_img.as_numpy_iterator().next()
# predictions = [np.argmax(pred) for pred in model.predict(image_batch)]
# predictions = model.predict_on_batch(image_batch).flatten()

predictions = model.predict(test_raw)
labels = []
for a,b in test_raw:
    labels.extend(b.numpy())
    # print(len(b.numpy()))
# print(predictions)
correct = {}
total = {}
for name in class_names:
    correct[name] = 0
    total[name] = 0
for (pred, b) in zip(predictions, labels):
    total[class_names[b]] = total[class_names[b]]+1
    if np.argmax(pred) == b:
        correct[class_names[b]] = correct[class_names[b]]+1
print("Total: ", total, "\nCorrect: ", correct)

# Apply a sigmoid since our model returns logits
# predictions = tf.nn.softmax(predictions)
# predictions = np.argmax(predictions)
# predictions = tf.where(predictions < 0.5, 0, 5)

# print('Predictions:\n', predictions)
# print('Labels:\n', label_batch)

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image_batch[i].astype("uint8"))
#     plt.title(class_names[predictions[i]])
#     plt.axis("off")

if __name__ == '__main__':
    print("end")
