import os
import csv
import cv2 as cv

print('Run with Python3')

training_input_path = input('Training images folder path (must end with /): ')
csv_train_path = input('Training ground truth table .csv file path: ')
test_input_path = input('Test images folder path (must end with /): ')
csv_test_path = input('Test ground truth table .csv file path: ')

output_path = '../data/task1/training_sorted_resized'
output_path_benign = output_path + '/benign'
output_path_malignant = output_path + '/malignant'

os.makedirs(output_path)
os.makedirs(output_path_benign)
os.makedirs(output_path_malignant)

with open(csv_train_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = '/' + row[0] + '.jpg'
        img_input_path = training_input_path + img_name
        img = cv.imread(img_input_path)
        resized = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)

        classification = row[1]
        
        if(classification == 'benign'):
            cv.imwrite(output_path_benign + img_name, resized)
        else:
            cv.imwrite(output_path_malignant + img_name, resized)

output_path = '../data/task1/test_sorted_resized'
output_path_benign = output_path + '/benign'
output_path_malignant = output_path + '/malignant'

os.makedirs(output_path)
os.makedirs(output_path_benign)
os.makedirs(output_path_malignant)

with open(csv_test_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = '/' + row[0] + '.jpg'
        img_input_path = test_input_path + img_name
        img = cv.imread(img_input_path)

        resized = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)

        classification = row[1]
    
        if(classification == '0.0'):
            cv.imwrite(output_path_benign + img_name, resized)
        else:
            cv.imwrite(output_path_malignant + img_name, resized)
