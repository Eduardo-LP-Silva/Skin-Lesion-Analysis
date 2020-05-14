import os
import csv
import shutil

print('Run with Python3')
training_input_path = input('Training images folder path (must end with /): ') #./resources/Task1/Training/
csv_input_path = input('Training ground truth table .csv file path: ') #./resources/Task1/Training-GT.csv
output_path = input('Output path (must end with /): ') #./resources/

output_path = output_path + 'Task1/TrainingSorted'
output_path_benign = output_path + '/benign'
output_path_malignant = output_path + '/malignant'

os.mkdir(output_path)
os.mkdir(output_path_benign)
os.mkdir(output_path_malignant)

with open(csv_input_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        img_name = '/' + row[0] + '.jpg'
        classification = row[1]
        #print(img_name + ' - ' + classification)
        if(classification == 'benign'):
            shutil.copyfile(training_input_path + img_name, output_path_benign + img_name)
        else:
            shutil.copyfile(training_input_path + img_name, output_path_malignant + img_name)