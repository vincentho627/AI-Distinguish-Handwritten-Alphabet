import os
import cv2
import csv


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
            path = os.path.join(PATH_TO_FILE, category)
            amount = 0
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    new_array //= 255
                    training_set = ""
                    for i in new_array:
                        training_set += ",".join([str(item) for item in (list(i))])
                        training_set += ','
                    writer.writerow([training_set, category])
                    amount += 1
                except Exception as e:
                    print(e)
            print("Category " + category + " has " + str(amount) + " pictures")
        file.close()

