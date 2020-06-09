import os
import csv
import shutil
import cv2 as cv
import math

TEST_FRACTION = 0.1

images_path = input('Training images folder path (must end with /): ')
csv_path = input('Training ground truth table .csv file path: ')

train_output_path = '../data/task2/training_sorted_resized'
test_outputh_path = '../data/task2/testing_sorted_resized'

os.makedirs(train_output_path)
os.makedirs(test_outputh_path)

classes = ['Melanoma', 'Melanocytic_Nevus', 'Basal_Cell_Carcinoma', 'Actinic_Keratosis', 'Benign_Keratosis', 'Dermatofibroma', 'Vascular_Lesion']

for class_name in classes:
    os.makedirs(train_output_path + '/' + class_name)
    os.makedirs(test_outputh_path + '/' + class_name)

with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    csv_reader.__next__()

    for row in csv_reader:
        img_name = row[0] + '.jpg'
        
        img_path = images_path + img_name
        img = cv.imread(img_path)
        print(img_path)
        resized_img = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)

        j = 1

        for j in range(1, len(row)):
            if row[j] == '1.0':
                classification = row[1]
                break
        
        print(train_output_path + '/' + classes[j - 1])
        cv.imwrite(train_output_path + '/' + classes[j - 1] + '/' + img_name, resized_img)

for directory in os.listdir(train_output_path):
    train_sub_directory = os.path.join(train_output_path, directory)
    test_sub_directory = os.path.join(test_outputh_path, directory)

    for _, _, images in os.walk(train_sub_directory):
        total_images = len(images)
        test_images_no = math.floor(total_images * TEST_FRACTION)

        for i in range(test_images_no):
            shutil.move(os.path.join(train_sub_directory, images[i]), os.path.join(test_sub_directory, images[i]))