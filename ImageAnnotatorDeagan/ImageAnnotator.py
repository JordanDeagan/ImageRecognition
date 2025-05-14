"""
ImageAnnotator - created by Muhammad T Karimi - 6/1/2020 - altered by Jordan Deagan - 6/26/2024
For the purpose of Annotating images in a streamlined and easy way. Designed so the user can annotate any number of
images contained in the images directory, and the choice of the user will be saved in
a csv file next to the image file name.
"""

import os
import datetime
import ctypes
import time
import random
import shutil
import cv2
import tkinter.filedialog as filedialog
import pathlib
import csv


# colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Global variables
todays_date = f"{datetime.datetime.today().month}-{datetime.datetime.today().day}-{datetime.datetime.today().year}"
while True:
    image_set_path = filedialog.askdirectory(title="Locate Image Set")
    if image_set_path is None or image_set_path == "":
        print("\nNo valid folder selected, please select a valid folder.\n")
    elif not os.path.isdir(f"{image_set_path}/ToAnnotate"):
        print("Folder does not contain \"ToAnnotate\" folder, please add folder with images to annotate")
    else:
        break

user = os.path.abspath("").split("\\")[2]  # current user, this value is used in the 4 variables below
curDir = os.getcwd()

path_to_imgs_to_annotate_dir = f"{image_set_path}/ToAnnotate"  # path to imgs not annotated
path_to_imgs_temp_dir = f"{image_set_path}/Temp"  # path to imgs being annotated
pathlib.Path(path_to_imgs_temp_dir).mkdir(parents=True, exist_ok=True)
path_to_imgs_annotated_dir = f"{image_set_path}/Annotated"  # path to imgs annotated
pathlib.Path(path_to_imgs_annotated_dir).mkdir(parents=True, exist_ok=True)
path_to_output_csv = f"{image_set_path}/output-{user}.csv"  # path to the CSV output file

with open(f"{curDir}/Categories.csv", "r", encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    categories = list(reader)
    primary_category = categories[0]
    listOfLists = []
    for num in range(1, len(categories)):
        listOfLists.append(categories[num])

# primary_category = {1: "Brick", 2: "Wood/Siding", 3: "Glass", 4: "Concrete", 5: "Steel", 6: "Stone", 7: "None",
#                     8: "Deleted"}
# nums_to_use_dict = {1: "Single Family Dwelling", 2: "Small Multi Family Dwelling - 2-4 Units",
#                     3: "Medium Multi Family Dwelling - 5-50 Units", 4: "Large Multi Family Dwelling - >50 Units",
#                     5: "Retail or Professional Services", 6: "Industry/Manufacturing", 7: "Technology/Research",
#                     8: "Entertainment & Recreation", 9: "Colleges/Universities", 10: "Grade Schools", 11: "Government",
#                     12: "Emergency Response", 13: "Hospital", 14: "Medical Office/Clinic", 15: "Nursing Home",
#                     16: "Mobile Home", 17: "Parking", 18: "Banks", 19: "Agriculture", 20: "Lodging",
#                     21: "Church/Non-Profit", 22: "Construction"}

"""
This function asks the user what the primary material of the building in the image is, and then asks the secondary 
material that makes up the building.
  - The user can select 'Delete' for the primary material, indicating that they believe that the image is not useful
  - The user can also select 'None' for the secondary material, meaning there is no or not enough of a 2nd material
    for it to be worth mentioning

@return tuple that has the primary material and the secondary material
"""


def ask_primary_category():
    while True:
        print("\nWhat category does this image belong to?")
        line = ""
        for catNum in range(len(primary_category)):
            if catNum % 2:
                line += f"\t\t{catNum + 1} {primary_category[catNum]}"
                print(line)
                line = ""
            else:
                line = f"\n{catNum + 1} {primary_category[catNum]}"
        if line != "":
            print(line)
        user_choice = input("Enter number: ")
        if user_choice == "" or user_choice.isalpha() or int(user_choice) < 1 or int(user_choice) > len(primary_category):
            print("Invalid choice")  # if user entered <1 or >8 or not a number
            continue
        break
    return int(user_choice)  # ignore errors, will never reach this point without values


"""
This function asks the user what percentage of the building is made up by the primary material in the image
  - If the user chose None or Delete for the image, then this question is skipped and N/A is printed in its place

@return the string with the answer (either a percentage or N/A)
"""

# def ask_percentage_material():
#     choices_arr = ["100%", "75%", "50%", "25%"]
#     while True:
#         print("\nWhat percentage of the building is made up by the primary material? Choose the closest answer."
#               "\n1. 100%\t\t2. 75%\t\t3. 50%\t4. 25%")
#         user_choice = input("Enter number: ")
#         try:
#             if user_choice.isalpha() or int(user_choice) < 1 or int(user_choice) > 4:
#                 print("Invalid choice")  # if user entered <1 or >4 or not a number
#                 continue
#         except (IndexError, ValueError):
#             print("Invalid choice")
#             continue
#         break
#     return choices_arr[int(user_choice) - 1]  # will never reach this point without values


"""
This function asks the user what the category of use of the building in the image is.

@return building_use the use of the building as selected by the user
"""


def ask_secondary_categories(category):
    temp_list = listOfLists[category]
    while True:
        print("\nWhat category does this image belong to?")
        line = ""
        for catNum in range(len(temp_list)):
            if catNum % 2:
                line += f"\t\t{catNum + 1} {temp_list[catNum]}"
                print(line)
                line = ""
            else:
                line = f"\n{catNum + 1} {temp_list[catNum]}"
        if line != "":
            print(line)
        user_choice = input("Enter number: ")
        try:
            if user_choice == "" or user_choice.isalpha() or int(user_choice) < 1 or int(user_choice) > len(temp_list):
                print("Invalid choice\n")  # if user entered < 1 or > 22 or not a number
                continue
        except ValueError:
            print("Invalid choice\n")
            continue
        break
    return int(user_choice)


"""
This function enables VT100 emulation, a Windows 10 setting that allows the color codes used above to actually
work and show the different colors. Otherwise, the colors would not work on the majority of terminals.
"""


def colors():
    kernel32 = ctypes.WinDLL('kernel32')
    hStdOut = kernel32.GetStdHandle(-11)
    mode = ctypes.c_ulong()
    kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode))
    mode.value |= 4
    kernel32.SetConsoleMode(hStdOut, mode)


# Call colors() to enable colors in text
colors()

# Start up and main menu
print(f"\n{bcolors.HEADER}WELCOME TO THE IMAGEANNOTATOR{bcolors.ENDC}")
print("The purpose of this software is to categorize images displayed to the user, the results of which"
      " will be written to\na CSV file. In order for the software to work properly, there are some rules and"
      "guidelines that must be followed.")
print("\nOnce you start, you will be asked how many images you want to annotate for this session. "
      "Then you will be shown images\none-by-one, and for each image you must:")
print("1. Determine the primary category the image belongs to and then enter that option into the system.")
print("2. Determine any additional category the image might belong to and enter that option into the system")

print(f"\n{bcolors.WARNING}Note:{bcolors.ENDC}")
print("  -Questions should appear 2 seconds after the image is shown. If a question does not appear, press any key\n\t"
      "while on the console.")
print("  -You also have the option of deleting an image if the content is difficult to see, covered by other objects,"
      " not a\n\tgood representation of that content, too big or too small, if there are too many other objects in the"
      " image, or\n\tfor other reasons you deem significant enough. If you are unsure, don't delete it.")
print("  -When you start a session, please finish the session completely, do not end halfway.")
print("  -The program will inform you when you have started and finished a session.")
print("  -The program will let you know how many images are left to annotate, so you can enter any number <= that.")

while True:
    answer = input("\nDo you acknowledge the above, and are ready to begin?\n"
                   "a. Yes\n"
                   "b. No\n\n"
                   "Enter your choice: ")
    if answer.lower() == 'b':
        quit(0)
    elif answer.lower() != 'a':
        print("Invalid choice")
    else:
        break

to_annotate = os.listdir(path_to_imgs_to_annotate_dir)
while True:
    imgs_left = len(to_annotate)
    print("\nNumber of images left to annotate: " + str(imgs_left))
    answer = input("How many images do you want to annotate: ")
    if not answer.isdigit() or int(answer) < 1 or int(answer) > imgs_left:
        print("Invalid choice")
    else:
        num_images = int(answer)
        break

# Quality Test section - asking user same questions on two images who's values are already known
# img = cv2.imread("quality_check_imgs/barn.jpg")  # show first test img
# cv2.imshow("Image 1", img)
# cv2.waitKey(2000)  # wait 2 seconds
# primary_material, secondary_material = ask_primary_category()  # ask user 2 main materials make up the building
# building_use = ask_secondary_categories()  # ask what category of use the building is used for
# cv2.destroyAllWindows()  # close the image
#
# if user got a question wrong, show them which one
# if primary_material != 2 or secondary_material != 7 or building_use != 19:
#     print(f"\n{bcolors.OKBLUE}This is a test question. Please review the correct answers.{bcolors.ENDC}")
#     print(f"For primary material, you said: {bcolors.FAIL}{primary_category[primary_material]}{bcolors.ENDC}"
#           "\t\tCorrect answer: Wood/Siding" if primary_material != 2 else "For primary material, you said: "
#                                                                           f"{primary_category[primary_material]}\t\tCorrect answer: Wood/Siding")
#     print(f"For secondary material, you said: {bcolors.FAIL}{primary_category[secondary_material]}"
#           f"{bcolors.ENDC}\t\tCorrect answer: None" if secondary_material != 7 else "For secondary material, you "
#                                                                                     f"said: {primary_category[secondary_material]}\t\tCorrect answer: None")
#     print(f"For building use, you said: {bcolors.FAIL}{nums_to_use_dict[building_use]}{bcolors.ENDC}"
#           f"\t\tCorrect answer: Agriculture" if building_use != 19 else
#           f"For building use, you said: {nums_to_use_dict[building_use]}\t\tCorrect answer: Agriculture")
#     time.sleep(5)  # pause system to give user time to read corrections before continuing
#
# img = cv2.imread("quality_check_imgs/office.jpg")  # show 2nd test img
# cv2.imshow("Image 2", img)
# cv2.waitKey(2000)
# primary_material, secondary_material = ask_primary_category()  # ask user 2 main materials make up the building
# building_use = ask_secondary_categories()
# cv2.destroyAllWindows()  # close the image
#
# # if user got a question wrong, show them which one
# if primary_material != 4 or secondary_material != 3 or building_use != 5:
#     print(f"\n{bcolors.OKBLUE}This is a test question. Please review the correct answers.{bcolors.ENDC}")
#     print(f"For primary material, you said: {bcolors.FAIL}{primary_category[primary_material]}{bcolors.ENDC}"
#           f"\t\tCorrect answer: Concrete" if primary_material != 4 else f"For primary material, you said: "
#                                                                         f"{primary_category[primary_material]}\t\tCorrect answer: Concrete")
#     print(f"For secondary material, you said: {bcolors.FAIL}{primary_category[secondary_material]}"
#           f"{bcolors.ENDC}\t\tCorrect answer: Glass" if secondary_material != 3 else f"For secondary material, "
#                                                                                      f"you said: {primary_category[secondary_material]}\t\tCorrect answer: Glass")
#     print(f"For building use, you said: {bcolors.FAIL}{nums_to_use_dict[building_use]}{bcolors.ENDC}"
#           f"\t\tCorrect answer: Retail or Professional Services" if building_use != 5 else f"For building use, "
#                                                                                            f"you said: {nums_to_use_dict[building_use]}\t\tCorrect answer: Retail or Professional Services")
#     time.sleep(5)  # pause system to give user time to read corrections before continuing

print(f"\n{bcolors.OKBLUE}Session start{bcolors.ENDC}")

images = []  # create list that will store the images to annotate for this session
for _ in range(num_images):  # iterate for the # of times = # of images user wanted to annotate
    try:
        image_to_move = random.choice(to_annotate)  # randomly choose an image from the ToAnnotate folder
        # Move it from ToAnnotate folder to the Temp folder, preventing any other instances from annotating the same img
        shutil.move(f"{path_to_imgs_to_annotate_dir}/{image_to_move}", f"{path_to_imgs_temp_dir}/{image_to_move}")
        images.append(image_to_move)  # add the moved image to the images to annotate list for this session
        to_annotate.remove(image_to_move)  # remove image from the to_annotate list, so that it isn't randomly chosen
    except:  # if any error occurs
        to_annotate = os.listdir(path_to_imgs_to_annotate_dir)  # update current list, and try again
        image_to_move = random.choice(to_annotate)  # same operations again
        shutil.move(f"{path_to_imgs_to_annotate_dir}/{image_to_move}", f"{path_to_imgs_temp_dir}/{image_to_move}")
        images.append(image_to_move)
        to_annotate.remove(image_to_move)

n = 1  # current image
num_images = len(images)  # total num images
for image in images:  # go over the images
    print(f"\n{bcolors.UNDERLINE}Image {n}/{num_images}{bcolors.ENDC}")  # print which image we're on

    try:  # check for any errors that might be thrown
        img = cv2.imread(f"{path_to_imgs_temp_dir}/{image}")  # start the image
        cv2.imshow(f"Image {n}", img)  # show the image
        cv2.waitKey(2000)  # waiting on user input (won't go away until user is done with image and moves to next)
    except:  # if error, most likely it is because image no longer exists
        print(f"{bcolors.WARNING}Image doesn't exist. Moving to next image.{bcolors.ENDC}")
        n += 1
        continue

    prime_category = ask_primary_category()  # ask user 2 main materials make up the building
    # if primary_material != 7 and primary_material != 8 and secondary_material != 8:
    #     material_percentage = ask_percentage_material()  # THIS WHOLE PART is only for asking percentage of material
    # else:
    #     material_percentage = "N/A"
    other_categories = []
    if len(listOfLists) > 0:
        for num in range(len(listOfLists)):
            other_categories.append(ask_secondary_categories(num))

    cv2.destroyAllWindows()  # close image

    i = 0
    while i < 10:  # writing to output.csv
        try:
            output_text = f"{image},{prime_category},"
            if len(listOfLists) > 0:
                for num in range(len(listOfLists)):
                    output_text = output_text + f"{other_categories[num]},"
            output_text = output_text + f"{primary_category[prime_category - 1]}"
            if len(listOfLists) > 0:
                for num in range(len(listOfLists)):
                    output_text = output_text + f",{listOfLists[num][other_categories[num] - 1]}"
            output_text = output_text + f"\n"
            with open(path_to_output_csv, "a") as output:  # write results to csv file, numbers first, then strings
                output.write(output_text)  # write the users choices to the CSV output file
            if i > 0:
                print(f"{bcolors.OKGREEN}Successful!{bcolors.ENDC}")
            break  # if it works, break the loop and continue with annotation
        except:  # if it doesn't work the first time, try again, up to 10 times
            print(f"{bcolors.WARNING}Writing to output.csv failed, trying again...{bcolors.ENDC}")
            i += 1

    if i < 10:  # if writing was successful, then move the image to the Annotated dir (which means its been annotated)
        shutil.move(f"{path_to_imgs_temp_dir}/{image}", f"{path_to_imgs_annotated_dir}/{image}")
    else:
        print(f"{bcolors.FAIL}Writing failed,{bcolors.OKBLUE} moving image back to ToAnnotate and progressing "
              f"to next image.{bcolors.ENDC}")
        shutil.move(f"{path_to_imgs_temp_dir}/{image}", f"{path_to_imgs_to_annotate_dir}/{image}")

    n += 1  # done with that image now

if num_images > 0:  # only print this if there were images chosen to be annotated
    print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}Session complete! Thank you for your time and help!{bcolors.ENDC}")
