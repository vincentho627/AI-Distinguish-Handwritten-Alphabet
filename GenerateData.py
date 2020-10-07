import os
import cv2
import csv
from DetectingLetter import *
import matplotlib.pyplot as plt

IMG_SIZE = 30
PATH_TO_FILE = os.path.join(os.getcwd(), "DATA")
CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']
headers = ",".join([str(item) for item in range(1, IMG_SIZE**2 + 1)] + ["class"])


def GetTrainingData(file_name):
    file_name = os.path.join(os.getcwd(), file_name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file, delimiter=" ")
        writer.writerow([headers])
        for category in CATEGORIES:
            # finds path for the image directories
            path = os.path.join(PATH_TO_FILE, category)
            amount = 0
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    # regularise the array by dividing all elements by 255
                    img_array //= 255
                    # resize the photo to 60x60 array to decrease cost of for loops in DetectingLetter() function
                    new_array = cv2.resize(img_array, (IMG_SIZE*2, IMG_SIZE*2))
                    # returns a sub-array of the img array which boxes the letter
                    boxed_array = DetectingLetter(new_array)
                    # resize to 30x30 array
                    boxed_array = cv2.resize(boxed_array, (IMG_SIZE, IMG_SIZE))
                    plt.imshow(boxed_array, cmap='gray')
                    plt.show()
                    training_set = ""
                    # writing the arrays onto the DATA.csv file
                    for i in boxed_array:
                        training_set += ",".join([str(item) for item in (list(i))])
                        training_set += ','
                    writer.writerow([training_set, category])
                    amount += 1
                except Exception as e:
                    print(e)
                    
            print("Category " + category + " has " + str(amount) + " pictures")
        file.close()


GetTrainingData('DATA.csv')

