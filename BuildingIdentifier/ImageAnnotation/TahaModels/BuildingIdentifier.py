# Much of the code is from the tensorflow tutorial site: https://www.tensorflow.org/tutorials/images/classification

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL

# Setting paths to training and validation datasets
# main_path = \
#     "C:/Users/mkarimi/OneDrive - Environmental Protection Agency (EPA)/ImageAnnotation/ConstructionMaterialDataset"
main_path = \
    "C:/Users/mkarimi/OneDrive - Environmental Protection Agency (EPA)/ImageAnnotation/Diby"
train_dir = os.path.join(main_path, "train")
validation_dir = os.path.join(main_path, "validation")

train_brick_dir = os.path.join(train_dir, "Brick")
train_woodsiding_dir = os.path.join(train_dir, "Wood_or_Siding")
train_glass_dir = os.path.join(train_dir, "Glass")
train_concrete_dir = os.path.join(train_dir, "Concrete")
train_steel_dir = os.path.join(train_dir, "Steel")
train_stone_dir = os.path.join(train_dir, "Stone")
#train_none_dir = os.path.join(train_dir, "None")
#train_deleted_dir = os.path.join(train_dir, "Deleted")

validation_brick_dir = os.path.join(validation_dir, "Brick")
validation_woodsiding_dir = os.path.join(validation_dir, "Wood_or_Siding")
validation_glass_dir = os.path.join(validation_dir, "Glass")
validation_concrete_dir = os.path.join(validation_dir, "Concrete")
validation_steel_dir = os.path.join(validation_dir, "Steel")
validation_stone_dir = os.path.join(validation_dir, "Stone")
#validation_none_dir = os.path.join(validation_dir, "None")
#validation_deleted_dir = os.path.join(validation_dir, "Deleted")

# Checking data
print("Brick training images: ", len(os.listdir(train_brick_dir)))
print("Wood/Siding training images: ", len(os.listdir(train_woodsiding_dir)))
print("Glass training images: ", len(os.listdir(train_glass_dir)))
print("Concrete training images: ", len(os.listdir(train_concrete_dir)))
print("Steel training images: ", len(os.listdir(train_steel_dir)))
print("Stone training images: ", len(os.listdir(train_stone_dir)))
#print("None training images: ", len(os.listdir(train_none_dir)))
#print("Deleted training images: ", len(os.listdir(train_deleted_dir)))
total_train = len(os.listdir(train_brick_dir)) + len(os.listdir(train_woodsiding_dir)) + len(os.listdir(train_glass_dir)) \
              + len(os.listdir(train_concrete_dir)) + len(os.listdir(train_steel_dir)) + len(os.listdir(train_stone_dir)) #\
              #+ len(os.listdir(train_none_dir)) + len(os.listdir(train_deleted_dir))

print("Brick validation images: ", len(os.listdir(validation_brick_dir)))
print("Wood/Siding validation images: ", len(os.listdir(validation_woodsiding_dir)))
print("Glass validation images: ", len(os.listdir(validation_glass_dir)))
print("Concrete validation images: ", len(os.listdir(validation_concrete_dir)))
print("Steel validation images: ", len(os.listdir(validation_steel_dir)))
print("Stone validation images: ", len(os.listdir(validation_stone_dir)))
#print("None validation images: ", len(os.listdir(validation_none_dir)))
#print("Deleted validation images: ", len(os.listdir(validation_deleted_dir)))
total_validation = len(os.listdir(validation_brick_dir)) + len(os.listdir(validation_woodsiding_dir)) + \
 len(os.listdir(validation_glass_dir)) + len(os.listdir(validation_concrete_dir)) + len(os.listdir(validation_steel_dir)) + \
 len(os.listdir(validation_stone_dir)) #+ len(os.listdir(validation_none_dir)) + len(os.listdir(validation_deleted_dir))

print("\nTotal train: ", total_train)
print("Total validation: ", total_validation)

# Memory management stuff
tf.keras.backend.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Setting constants
batch_size = 32
epochs = 150
img_height = 154 # 256
img_width = 154 # 256
IMG_SIZE = (img_width, img_height)

""" METHOD 1 - TRYING TO TRAIN USING MY OWN MODEL

# Data preparation and loading images
train_img_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15, 
                zoom_range=0.4, width_shift_range=.15, height_shift_range=.15)
validation_img_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = train_img_gen.flow_from_directory(batch_size=batch_size, directory=train_dir,
                                                   shuffle=True, target_size=(img_height, img_width),
                                                   class_mode='categorical')
validation_data_gen = validation_img_gen.flow_from_directory(batch_size=batch_size, directory=validation_dir,
                                                   shuffle=True, target_size=(img_height, img_width),
                                                   class_mode='categorical')

# Visualize some of the training images using matplotlib
sample_training_images, _ = next(train_data_gen)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plotImages(sample_training_images[:5])

# Model Creation
model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=2, strides=1, padding='same'),  # also have 'valid'
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2, strides=1, padding='same'),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2, strides=1, padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),  # also have 'tanh' and 'sigmoid'
    Dense(6)  # activation can also be 'softmax'
])
globals()['_model'] = model

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])  # can also do sparse_categorical_crossentropy

# another compilation option
""""""
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
""""""

# Summary of the model
model.summary()

# Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train//batch_size,  # batch size
    epochs=epochs,
    validation_data=validation_data_gen,
    validation_steps=total_validation//batch_size  # batch size for validation
)

# with open("trainingResults.txt", "w") as results:
#     results.write(f"{history.history['accuracy']}\n{history.history['val_accuracy']}")

# # Visualize the results
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

"""

""" METHOD 2 - TRYING TO TRAIN USING TRANSFER LEARNING (MobileNetV2) """
# Data preparation and loading images
train_img_gen = ImageDataGenerator(rescale=1./127.5, horizontal_flip=True, rotation_range=15, 
                zoom_range=0.4, width_shift_range=.15, height_shift_range=.15)
validation_img_gen = ImageDataGenerator(rescale=1./127.5)

train_data_gen = train_img_gen.flow_from_directory(batch_size=batch_size, directory=train_dir,
                                                   shuffle=True, target_size=(img_height, img_width),
                                                   class_mode='categorical')
validation_data_gen = validation_img_gen.flow_from_directory(batch_size=batch_size, directory=validation_dir,
                                                   shuffle=True, target_size=(img_height, img_width),
                                                   class_mode='categorical')
# Get the pre-trained model MobileNetV2, now using EfficientNetB7
IMG_SHAPE = IMG_SIZE + (3,)
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.EfficientNetB7(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.EfficientNetB6(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.ResNet152V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model = tf.keras.applications.DenseNet201(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

image_batch, label_batch = next(iter(train_data_gen))
feature_batch = base_model(image_batch)
print(feature_batch.shape)  # take a look at the shape of a batch

base_model.trainable = False  # freeze the conv base

base_model.summary()

# convert features to a single 1280? element vector per image
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# layer that does the predicting
prediction_layer = tf.keras.layers.Dense(6)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Build the model
inputs = tf.keras.Input(shape=(154, 154, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(train_data_gen, epochs=epochs, validation_data=validation_data_gen)

# FINE TUNE THE MODEL
base_model.trainable = True  # unfreeze layers

print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 450  # fine tune from this layer onwards
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10), metrics=['accuracy'])
model.summary()

# Start the fine tuning
history_fine_tuning = model.fit(train_data_gen, epochs=epochs, initial_epoch=history.epoch[-1], validation_data=validation_data_gen)
