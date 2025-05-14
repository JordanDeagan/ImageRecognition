# Utility file developed by Taha Karimi
# Helps automate various data organizing/adjusting/editing functions

import shutil, sys, os
from PIL import Image
import numpy as np
from numpy import asarray

path_to_annotated = "/Annotated"
path_to_temp = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/Temp"
path_to_toannotate = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/ToAnnotate"
path_to_constr = \
    "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/ConstructionMaterialDataset"
path_to_output_file = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/output.csv"
path_to_final_output_file = "C:/Users/mkarimi/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/BuildingIdentifier/ImageAnnotator/QC_Boe/output.csv"
path_to_output_QC_file = "C:/Users/mkarimi/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/BuildingIdentifier/ImageAnnotator/QC_Boe/output-tboe.csv"
path_to_final_output_QC_file = "C:/Users/mkarimi/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/BuildingIdentifier/ImageAnnotator/QC_Boe/outputoriginalQC.csv"

path_to_keras_tuner_train_images = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/KerasTunerTrainImages"
path_to_keras_tuner_test_images = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/KerasTunerTestImages"

path_to_keras_tuner_train = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/KerasTunerTrain"
path_to_keras_tuner_test = "C:/Users/jdeagan/OneDrive - Environmental Protection Agency (EPA)/Profile/Desktop/Projects/AI/ImageAnnotation/KerasTunerTest"

num_to_move = 9657
num_train = 8692

def move_from_annotated_to_trainandvalidation():
    with open(path_to_output_file, "r") as csv:
        for i in range(num_to_move):
            line = csv.readline()
            line_array = line.split(",")
            print(i, line_array)
            file_name = line_array[0]
            primary_material = line_array[4]
            if primary_material == "Wood/Siding":
                primary_material = "Wood_or_Siding"
            if i < num_train:  # then move to train folder
                shutil.copy(f"{path_to_annotated}/{file_name}", f"{path_to_constr}/train/{primary_material}")
            else:  # else move to validation folder
                shutil.copy(f"{path_to_annotated}/{file_name}", f"{path_to_constr}/validation/{primary_material}")

def move_from_temp_to_toannotate():
    file_names = os.listdir(path_to_temp)
    i = 0
    for file_name in file_names:
        shutil.move(f"{path_to_temp}/{file_name}", f"{path_to_toannotate}")
        i += 1
    print(f"Moved {i} images.")

def remove_duplicates_from_toannotate():
    annotated_imgs = os.listdir(path_to_annotated)
    i = 0
    for image_name in annotated_imgs:
        try:
            os.remove(f"{path_to_toannotate}/{image_name}")
            i += 1
        except FileNotFoundError:
            continue
    print(f"Removed {i} images.")

def move_undocumented_images_back_to_toannotate():
    csv_images = []  # array to contain documented images
    with open(path_to_output_file, "r") as csv:
        for line in csv:
            line_array = line.split(",")
            csv_images.append(line_array[0])  # add in documented images

    annotated_images = os.listdir(path_to_annotated)  # get all images that are in the Annotated folder
    i = 0
    for image in annotated_images:
        if image not in csv_images:
            shutil.move(f"{path_to_annotated}/{image}", f"{path_to_toannotate}")
            i += 1
    print(f"Moved {i} images.")

def clear_train_val_folders():
    train_path = path_to_constr + "/train"
    val_path = path_to_constr + "/validation"

    materials = ["/Brick", "/Concrete", "/Deleted", "/Glass", "/None", "/Steel", "/Stone", "/Wood_or_Siding"]

    # count the number that will be deleted in train and val
    total_train_exp = os.listdir(train_path + "/Brick") + os.listdir(train_path + "/Concrete") + os.listdir(train_path + "/Deleted") \
        + os.listdir(train_path + "/Glass") + os.listdir(train_path + "/None") + os.listdir(train_path + "/Steel") + \
        os.listdir(train_path + "/Stone") + os.listdir(train_path + "/Wood_or_Siding")

    total_val_exp = os.listdir(val_path + "/Brick") + os.listdir(val_path + "/Concrete") + os.listdir(val_path + "/Deleted") \
        + os.listdir(val_path + "/Glass") + os.listdir(val_path + "/None") + os.listdir(val_path + "/Steel") + \
        os.listdir(val_path + "/Stone") + os.listdir(val_path + "/Wood_or_Siding")

    total_train = 0
    total_val = 0

    # delete training images
    for material in materials:
        material_path_to_delete = train_path + f"/{material}"
        imgs_to_delete = os.listdir(material_path_to_delete)

        for img in imgs_to_delete:
            os.remove(material_path_to_delete + f"/{img}")
            total_train += 1

    # delete val images
    for material in materials:
        material_path_to_delete = val_path + f"/{material}"
        imgs_to_delete = os.listdir(material_path_to_delete)

        for img in imgs_to_delete:
            os.remove(material_path_to_delete + f"/{img}")
            total_val += 1

    print(f"Total train expected to be deleted: {len(total_train_exp)}")
    print(f"Total val expected to be deleted: {len(total_val_exp)}")
    print(f"Total train deleted: {total_train}")
    print(f"Total val deleted: {total_val}")

def count_train_val_imgs():
    train_path = path_to_constr + "/train"
    val_path = path_to_constr + "/validation"

    # count and print the number of imgs in each folder
    total_brick = len(os.listdir(train_path + "/Brick"))
    total_concrete = len(os.listdir(train_path + "/Concrete"))
    total_deleted = len(os.listdir(train_path + "/Deleted"))
    total_glass = len(os.listdir(train_path + "/Glass"))
    total_none = len(os.listdir(train_path + "/None"))
    total_steel = len(os.listdir(train_path + "/Steel"))
    total_stone = len(os.listdir(train_path + "/Stone"))
    total_woodorsiding = len(os.listdir(train_path + "/Wood_or_Siding"))
    total_train_exp = total_brick + total_concrete + total_deleted + total_glass + total_none + total_steel + \
        total_stone + total_woodorsiding

    print(f"Train Brick: {total_brick}")
    print(f"Train Concrete: {total_concrete}")
    print(f"Train Deleted: {total_deleted}")
    print(f"Train Glass: {total_glass}")
    print(f"Train None: {total_none}")
    print(f"Train Steel: {total_steel}")
    print(f"Train Stone: {total_stone}")
    print(f"Train Wood/Siding: {total_woodorsiding}")
    print(f"Total Train: {total_train_exp}")

    total_val_brick = len(os.listdir(val_path + "/Brick"))
    total_val_concrete = len(os.listdir(val_path + "/Concrete"))
    total_val_deleted = len(os.listdir(val_path + "/Deleted"))
    total_val_glass = len(os.listdir(val_path + "/Glass"))
    total_val_none = len(os.listdir(val_path + "/None"))
    total_val_steel = len(os.listdir(val_path + "/Steel"))
    total_val_stone = len(os.listdir(val_path + "/Stone"))
    total_val_woodorsiding = len(os.listdir(val_path + "/Wood_or_Siding"))
    total_val_exp = total_val_brick + total_val_concrete + total_val_deleted + total_val_glass + total_val_none + total_val_steel \
     + total_val_stone + total_val_woodorsiding

    print(f"\nVal Brick: {total_val_brick}")
    print(f"Val Concrete: {total_val_concrete}")
    print(f"Val Deleted: {total_val_deleted}")
    print(f"Val Glass: {total_val_glass}")
    print(f"Val None: {total_val_none}")
    print(f"Val Steel: {total_val_steel}")
    print(f"Val Stone: {total_val_stone}")
    print(f"Val Wood/Siding: {total_val_woodorsiding}")
    print(f"Total Val: {total_val_exp}")

def qccopytocreateoriginal():
    QCd_array = []
    with open(path_to_output_QC_file, "r") as qc_csv:
        total_qc = 0
        for i in range(100):
            line = qc_csv.readline()
            line_array = line.split(",")
            total_qc += 1
            QCd_array.append(line_array[0])
    print(f"Total QC: {total_qc}")
    final_output_QC_array = []

    total_matched = 0
    for i in range(len(QCd_array)):
        with open(path_to_final_output_file, "r") as csv:
            for j in range(num_to_move):
                line = csv.readline()
                line_array = line.split(",")
                if line_array[0] == QCd_array[i]:
                    final_output_QC_array.append(line)
                    total_matched += 1
  # with open(path_to_final_output_file, "r") as csv:
        # for i in range(num_to_move):
        #     line = csv.readline()
        #     line_array = line.split(",")
        #     for j in range(len(QCd_array)):
        #         if line_array[0] == QCd_array[j]:
        #             final_output_QC_array.append(line)
        #             total_matched += 1
    print(f"Total matched: {total_matched}")
    with open(path_to_final_output_QC_file, "a") as final_qc_csv:
        total_written = 0
        for line in final_output_QC_array:
            final_qc_csv.write(line)
            total_written += 1
    print(f"Total written: {total_written}")

def convert_to_png():
    train_folders = os.listdir(path_to_keras_tuner_train)
    test_folders = os.listdir(path_to_keras_tuner_test)

    for train_folder in train_folders:
        images = os.listdir(path_to_keras_tuner_train + "/" + train_folder)

        for image in images:
            newImage = Image.open(path_to_keras_tuner_train + "/" + train_folder + "/" + image)
            # data = asarray(newImage)
            # print(data)
            newImageName = image[:len(image) - 4] + ".png"
            newImage.save(path_to_keras_tuner_train + "/" + train_folder + "/" + newImageName)

    for test_folder in test_folders:
        images = os.listdir(path_to_keras_tuner_test + "/" + test_folder)

        for image in images:
            newImage = Image.open(path_to_keras_tuner_test + "/" + test_folder + "/" + image)
            # data = asarray(newImage)
            # print(data)
            newImageName = image[:len(image) - 4] + ".png"
            newImage.save(path_to_keras_tuner_test + "/" + test_folder + "/" + newImageName)

def create_image_and_label_list():
    arrayOfTrainImages = os.listdir(path_to_keras_tuner_train_images)
    arrayOfTestImages = os.listdir(path_to_keras_tuner_test_images)

    # FOR TRAINING IMAGES/LABELS
    # Now create the dictionary by iterating through the training folders
    dict_train_im_labels = {}
    
    list_of_directories = os.listdir(path_to_keras_tuner_train)

    for material in list_of_directories:  # add all train images and corresponding labels to a dictionary
        images_in_directory = os.listdir(path_to_keras_tuner_train + "/" + material)
        
        for image in images_in_directory:  # 0=Brick, 1=Concrete, 2=Glass, 3=Steel, 4=Stone, 5=Wood_or_Siding
            label = 50  # random large number showing that something went wrong
            if material == "Brick":
                label = 0
            elif material == "Concrete":
                label = 1
            elif material == "Glass":
                label = 2
            elif material == "Steel":
                label = 3
            elif material == "Stone":
                label = 4
            elif material == "Wood_or_Siding":
                label = 5
            
            dict_train_im_labels[image] = label
        
    # Create array of training labels in same order as the array of images in Annotated directory
    array_of_train_labels = []
    for image in arrayOfTrainImages:
        array_of_train_labels.append(dict_train_im_labels[image])

    # Print this stuff to a file
    train_images_csv = open("../train_images.csv", "a")
    train_labels_csv = open("../train_labels.csv", "a")

    i = 0
    for image in arrayOfTrainImages:
        train_images_csv.write(image + "\n")
        train_labels_csv.write(f"{array_of_train_labels[i]}\n")
        i += 1

    train_images_csv.close()
    train_labels_csv.close()
    
    # FOR TEST/VAL IMAGES/LABELS
    # Now create the dictionary by iterating through the validation folders
    dict_test_im_labels = {}
    
    list_of_directories = os.listdir(path_to_keras_tuner_test)

    for material in list_of_directories:  # add all val images and corresponding labels to a dictionary
        images_in_directory = os.listdir(path_to_keras_tuner_test + "/" + material)
        
        for image in images_in_directory:  # 0=Brick, 1=Concrete, 2=Glass, 3=Steel, 4=Stone, 5=Wood_or_Siding
            label = 51  # random large number showing that something went wrong, 51 for val
            if material == "Brick":
                label = 0
            elif material == "Concrete":
                label = 1
            elif material == "Glass":
                label = 2
            elif material == "Steel":
                label = 3
            elif material == "Stone":
                label = 4
            elif material == "Wood_or_Siding":
                label = 5
            
            dict_test_im_labels[image] = label
        
    # Create array of training labels in same order as the array of images in Annotated directory
    array_of_test_labels = []
    for image in arrayOfTestImages:
        array_of_test_labels.append(dict_test_im_labels[image])

    # Print this stuff to a file
    test_images_csv = open("../test_images.csv", "a")
    test_labels_csv = open("../test_labels.csv", "a")

    i = 0
    for image in arrayOfTestImages:
        test_images_csv.write(image + "\n")
        test_labels_csv.write(f"{array_of_test_labels[i]}\n")
        i += 1

    test_images_csv.close()
    test_labels_csv.close()

def copy_images_to_resize():
    ### MOVING THE TEST IMAGES
    """
    test_images = os.listdir(path_to_keras_tuner_test_images)
    for image in test_images:
        if (np.array(Image.open(f"{path_to_keras_tuner_test_images}/{image}"))).shape != (256, 256, 3):
            shutil.move(f"{path_to_keras_tuner_test_images}/{image}", "TempTest")

    soon_to_be_resized_images = os.listdir("TempTest")
    for image in soon_to_be_resized_images:
        new_image = (Image.open(f"TempTest/{image}")).resize((256, 256))
        new_image.save(f"TempTest2/{image}")
    """

    ### MOVING THE TRAIN IMAGES
    train_images = os.listdir(path_to_keras_tuner_train_images)
    for image in train_images:
        if (np.array(Image.open(f"{path_to_keras_tuner_train_images}/{image}"))).shape != (256, 256, 3):
            shutil.move(f"{path_to_keras_tuner_train_images}/{image}", "TempTrain")

    soon_to_be_resized_images = os.listdir("TempTrain")
    for image in soon_to_be_resized_images:
        new_image = (Image.open(f"TempTrain/{image}")).resize((256, 256))
        new_image.save(f"TempTrain2/{image}")


if __name__ == '__main__':
    try:
        function = sys.argv[1]
    except IndexError:
        print("Usage: python ProcessData.py <name of function to execute>" \
            "\nFunctions:" \
            "\n\tmovefortraining" \
            "\n\temptytemp" \
            "\n\tremoveduplicates" \
            "\n\tmoveundocumented" \
            "\n\tcleartrainval" \
            "\n\tcounttrainval" \
            "\n\tqccopytocreateoriginal" \
            "\n\tconverttopng" \
            "\n\tcopyimagestoresize")
        exit(0)

    if function == 'movefortraining':
        move_from_annotated_to_trainandvalidation()
    elif function == 'emptytemp':
        move_from_temp_to_toannotate()
    elif function == 'removeduplicates':
        remove_duplicates_from_toannotate()
    elif function == 'moveundocumented':
        move_undocumented_images_back_to_toannotate()
    elif function == 'cleartrainval':
        clear_train_val_folders()
    elif function == 'counttrainval':
        count_train_val_imgs()
    elif function == 'qccopytocreateoriginal':
        qccopytocreateoriginal()
    elif function == 'converttopng':
        convert_to_png()
    elif function == 'createimageandlabellist':
        create_image_and_label_list()
    elif function == 'copyimagestoresize':
        copy_images_to_resize()
