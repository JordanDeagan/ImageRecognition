import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import gc
import os

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

from keras_tuner.tuners import Hyperband
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel

print("THIS IS THE PROPER KERAS TUNER ///////////////////////////////////////////////////////////"
      "////////////////////////////////////////////////////////////////////////////////////////////////////////")
keras.backend.clear_session()  # these 2 are an attempt to fix resourceexchausted errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''

image_annotation = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/" \
                   "ArtificialIntel/ImageAnnotation/"
# train_dir = image_annotation + "KerasTunerTrain"
# test_dir = image_annotation + "KerasTunerTest"
# flower_dir = image_annotation + "flower_photos"
# cats_dogs_train = image_annotation + "cats_and_dogs_filtered/train"
# cats_dogs_val = image_annotation + "cats_and_dogs_filtered/validation"
train_dir = image_annotation + "KerasTunerReducedTrain"
test_dir = image_annotation + "KerasTunerReducedTest"

batch_size = 32
img_height = 256
img_width = 256
val_split = 0.1
# flower_split = 0.3
seed = 3

print(tf.__version__)

train_img = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_img = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_img = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# train_img = tf.keras.utils.image_dataset_from_directory(
#     cats_dogs_train,
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
#
# val_img = tf.keras.utils.image_dataset_from_directory(
#     cats_dogs_val,
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

# train_img = tf.keras.utils.image_dataset_from_directory(
#     flower_dir,
#     validation_split=flower_split,
#     subset="training",
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
#
# val_img = tf.keras.utils.image_dataset_from_directory(
#     flower_dir,
#     validation_split=flower_split,
#     subset="validation",
#     seed=seed,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

# val_batches = tf.data.experimental.cardinality(val_img)
# test_img = val_img.take(val_batches // 5)
# validation_dataset = val_img.skip(val_batches // 5)

class_names = train_img.class_names

# train_image_names = []
# train_images = []
# train_labels = []
# train_temp = []
#
# test_image_names = []
# test_images = []
# test_labels = []
# test_temp = []
#
# # tempArray = np.zeros((7735, 256, 256, 3), dtype=int, order='C')
#
# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#
# # Load the image names from the file in the correct order, as well as the labels in the correct
# # and thus corresponding order
# with open("train_images.csv", "r") as train_image_names_file:
#     # Read and print the entire file line by line
#     line = train_image_names_file.readline()
#     while line != '':  # The EOF char is an empty string
#         line = line[:len(line) - 1]
#         train_image_names.append(line)
#         line = train_image_names_file.readline()
# i = 0
# with open("train_labels.csv", "r") as train_labels_file:
#     # Read and print the entire file line by line
#     line = train_labels_file.readline()
#     while line != '':  # The EOF char is an empty string
#         label_arr = int(line[:len(line) - 1])
#         if label_arr == 3 or label_arr == 5:
#             train_labels.append(label_arr)
#             train_temp.append(train_image_names[i])
#         line = train_labels_file.readline()
#         i += 1
# train_image_names = train_temp
# train_labels = np.array(train_labels)
#
# # Now take the image names array and get the actual images, and read them in using PIL Image library
# # then store them in another array
# for i in range(int(len(train_image_names))):
#     train_images.append(np.array(Image.open(image_annotation + f"KerasTunerTrainImages/{train_image_names[i]}")))
# train_images = np.asarray(train_images)
#
# # i=1
# # print(Image.open(image_annotation + f"KerasTunerTrainImages/{train_image_names[i]}"))
#
# # Load the image names from the file in the correct order, as well as the labels in the correct
# # and thus corresponding order
# with open("test_images.csv", "r") as test_image_names_file:
#     # Read and print the entire file line by line
#     line = test_image_names_file.readline()
#     while line != '':  # The EOF char is an empty string
#         line = line[:len(line) - 1]
#         test_image_names.append(line)
#         line = test_image_names_file.readline()
#
# i = 0
#
# with open("test_labels.csv", "r") as test_labels_file:
#     # Read and print the entire file line by line
#     line = test_labels_file.readline()
#     while line != '':  # The EOF char is an empty string
#         label_arr = int(line[:len(line) - 1])
#         if label_arr == 3 or label_arr == 5:
#             test_labels.append(label_arr)
#             test_temp.append(test_image_names[i])
#         line = test_labels_file.readline()
#         i += 1
#
# test_labels = np.array(test_labels)
# test_image_names = test_temp
#
# # Now take the image names array and get the actual images, and read them in using PIL Image library
# # then store them in another array
# for i in range(int(len(test_image_names))):
#     test_images.append(np.array(Image.open(image_annotation + f"KerasTunerTestImages/{test_image_names[i]}")))
# test_images = np.asarray(test_images)
#
# # Put the train images into the other tempArray to make it a real numpy array
# # i = 0
# # for image in train_images:
# #   tempArray[i] = image
# #   i += 1
#
# x_train = train_images
# y_train = train_labels
# x_test = test_images
# y_test = test_labels
#
# # Pre-processing
# x_train = x_train.astype('int') / 255.
# x_test = x_test.astype('int') / 255.
#
# # print("x_test: %s\ny_test: %s" % (len(x_test), len(y_test)))
#
# AUTOTUNE = tf.data.AUTOTUNE

# x_train = x_train.cache().prefetch(buffer_size=AUTOTUNE)
# x_test = x_test.cache().prefetch(buffer_size=AUTOTUNE)

# print(x_test[0])


AUTOTUNE = tf.data.AUTOTUNE
train_img = train_img.cache().prefetch(buffer_size=AUTOTUNE)
val_img = val_img.cache().prefetch(buffer_size=AUTOTUNE)
test_img = test_img.cache().prefetch(buffer_size=AUTOTUNE)

INPUT_SHAPE = (img_height, img_width, 3)  # should be 415, 415, 3?
NUM_CLASSES = 6

"""
    Future Attempts:
        only 2 Conv
        only 1 Dense
        2 conv + 1 Dense
        remove batch + pooling between conv's
"""

class CNNHyperModelDeagan(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(RandomFlip("horizontal_and_vertical", input_shape=INPUT_SHAPE)),
        model.add(
            RandomRotation(
                # factor=hp.Float('rand1', min_value=0.1, max_value=0.55, default=0.2, step=0.1)
                factor=0.2
            )
        ),
        model.add(
            RandomZoom(
                # height_factor=hp.Float('rand2', min_value=0.1, max_value=0.5, default=0.2, step=0.1)
                height_factor=0.4
            )
        ),
        model.add(Rescaling(1. / 255)),
        model.add(
            Conv2D(filters=hp.Int('num_filters1', min_value=16, max_value=128, step=16, default=16),
                   kernel_size=hp.Choice('kernel1', values=[3, 4, 5], default=4),
                   activation=hp.Choice('conv_activation1', values=['relu', 'tanh', 'sigmoid'], default='relu')
                   # input_shape=self.input_shape
                   )
            # Conv2D(filters=64,
            #        kernel_size=4,
            #        activation='relu'
            #        # input_shape=self.input_shape
            #        )
        )
        model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        model.add(
            Conv2D(filters=hp.Int('num_filters2', min_value=16, max_value=128, step=16, default=32),
                   kernel_size=hp.Choice('kernel2', values=[3, 4, 5], default=4),
                   activation=hp.Choice('conv_activation2', values=['relu', 'tanh', 'sigmoid'], default='relu'))
            # Conv2D(filters=128,
            #        kernel_size=4,
            #        activation='relu'
            #        # input_shape=self.input_shape
            #        )
        )
        model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        model.add(
            Dropout(
                rate=hp.Float('dropout1', min_value=0.1, max_value=0.6, default=0.3, step=0.1)
                # rate=0.5
            )
        )
        model.add(
            Conv2D(filters=hp.Int('num_filters3', min_value=16, max_value=128, step=16, default=64),
                   kernel_size=hp.Choice('kernel3', values=[3, 4, 5], default=4),
                   activation=hp.Choice('conv_activation3', values=['relu', 'tanh', 'sigmoid'], default='relu'))
            # Conv2D(filters=256,
            #        kernel_size=4,
            #        activation='relu'
            #        # input_shape=self.input_shape
            #        )
        )
        model.add(MaxPooling2D())
        # model.add(BatchNormalization())
        model.add(
            Dropout(
                rate=hp.Float('dropout2', min_value=0.1, max_value=0.6, default=0.3, step=0.1)
                # rate=0.5
            )
        )
        model.add(Flatten())
        model.add(
            Dense(units=hp.Int('units1', min_value=96, max_value=284, step=16, default=128),
                  activation=hp.Choice('dense_activation1', values=['tanh', 'sigmoid', 'relu'], default='sigmoid'))
            # Dense(units=256,
            #       activation='tanh')
        )
        model.add(
            Dropout(
                rate=hp.Float('dropout3', min_value=0.1, max_value=0.6, default=0.3, step=0.1)
                # rate=0.5
            )
        )
        model.add(
            Dense(units=hp.Int('units2',  min_value=96, max_value=284, step=16, default=128),
                  activation=hp.Choice('dense_activation2', values=['tanh', 'sigmoid', 'relu'], default='sigmoid'))
            # Dense(units=416,
            #       activation='sigmoid')
        )
        # model.add(
        #     Dropout(
        #         rate=hp.Float('dropout2', min_value=0.0, max_value=0.2, default=0.1, step=0.1)
        #         # rate=0.2
        #     )
        # )
        model.add(
            Dense(self.num_classes, activation='softmax')
        )
        model.compile(
            # optimizer='adam',
            optimizer=keras.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-5, max_value=1e-4, sampling='LOG', default=2e-5)),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        return model


hypermodel = CNNHyperModelDeagan(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

HYPERBAND_MAX_EPOCHS = 50
FACTOR = 4
EXECUTION_PER_TRIAL = 1
SEED = 3
# MAX_TRIALS = 20
# N_EPOCH_SEARCH = 40
# BATCH_SIZE = 100
ES_PATIENCE = 3

my_callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,
        restore_best_weights=True)
]

tuner = None

try:
    tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        factor=FACTOR,
        objective='val_accuracy',
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        overwrite=True,
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

tuner.search(x=train_img,
             callbacks=my_callbacks,
             # batch_size=BATCH_SIZE,
             # epochs=N_EPOCH_SEARCH,
             validation_data=val_img)

tuner.results_summary()

bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]

best_model = tuner.get_best_models(num_models=1)[0]  # fails here for some reason
loss, accuracy = best_model.evaluate(x=test_img)

predictions = best_model.predict(test_img)
labels = []
for a,b in test_img:
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

# Some cleaning up
f = open("results.txt", "a")
f.write(
    # f"optimal rate in random_rotation layer: {bestHP.get('rand1')}\n"
    # f"optimal rate in random_zoom layer: {bestHP.get('rand2')}\n"
    f"optimal number of filters in conv_1 layer: {bestHP.get('num_filters1')}\n"
    f"optimal number of kernels in conv_1 layer: {bestHP.get('kernel1')}\n"
    f"optimal activation routine in conv_1 layer: {bestHP.get('conv_activation1')}\n"
    f"optimal number of filters in conv_2 layer: {bestHP.get('num_filters2')}\n"
    f"optimal number of kernels in conv_2 layer: {bestHP.get('kernel2')}\n"
    f"optimal activation routine in conv_2 layer: {bestHP.get('conv_activation2')}\n"
    f"optimal rate in dropout_1 layer: {bestHP.get('dropout1')}\n"
    f"optimal number of filters in conv_3 layer: {bestHP.get('num_filters3')}\n"
    f"optimal number of kernels in conv_3 layer: {bestHP.get('kernel3')}\n"
    f"optimal activation routine in conv_3 layer: {bestHP.get('conv_activation3')}\n"
    f"optimal rate in dropout_2 layer: {bestHP.get('dropout2')}\n"
    f"optimal number of units in dense_1 layer: {bestHP.get('units1')}\n"
    f"optimal activation routine in dense_1 layer: {bestHP.get('dense_activation1')}\n"
    f"optimal rate in dropout_3 layer: {bestHP.get('dropout3')}\n"
    f"optimal number of units in dense_2 layer: {bestHP.get('units2')}\n"
    f"optimal activation routine in dense_2 layer: {bestHP.get('dense_activation2')}\n"
    # f"optimal learning rate: {bestHP.get('learning_rate')}\n\n"
    f"Loss: {loss}\nAccuracy: {accuracy}\n\n")
f.close()

gc.collect()
del hypermodel
del tuner

if __name__ == '__main__':
    print("end")
