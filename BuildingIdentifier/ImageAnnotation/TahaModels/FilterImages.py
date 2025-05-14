import numpy as np
from PIL import Image

""" THIS SECTION IS FOR TRAINING IMAGES
train_image_names = []
train_images = []

# Load the image names from the file in the correct order, as well as the labels in the correct and thus corresponding order
with open("train_images.csv", "r") as train_image_names_file:
  # Read and print the entire file line by line
  line = train_image_names_file.readline()
  while line != '':  # The EOF char is an empty string
    line = line[:len(line) - 1]
    train_image_names.append(line)
    line = train_image_names_file.readline()

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for image_name in train_image_names:
  train_images.append(np.array(Image.open(f"KerasTunerTrainImages/{image_name}")))

print(len(train_images[0].shape))

i = 0
for image in train_images:
    if len(image.shape) != 3:
        print(i, image.shape, train_image_names[i])
    i += 1
"""

# THIS SECTION IS FOR TEST IMAGES
test_image_names = []
test_images = []

# Load the image names from the file in the correct order, as well as the labels in the correct and thus corresponding order
with open("../test_images.csv", "r") as test_image_names_file:
  # Read and print the entire file line by line
  line = test_image_names_file.readline()
  while line != '':  # The EOF char is an empty string
    line = line[:len(line) - 1]
    test_image_names.append(line)
    line = test_image_names_file.readline()

# Now take the image names array and get the actual images, and read them in using PIL Image library then store them in another array
for image_name in test_image_names:
  test_images.append(np.array(Image.open(f"KerasTunerTestImages/{image_name}")))

print(len(test_images[0].shape))

i = 0
for image in test_images:
    if len(image.shape) != 3:
        print(i, image.shape, test_image_names[i])
    i += 1
