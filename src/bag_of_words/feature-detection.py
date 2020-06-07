import sys
import csv
import random
import argparse
from shared import *
from sklearn.metrics import confusion_matrix
sys.path.append('..')


parser = argparse.ArgumentParser(description='VCOM - Bag of Words Tumor classification')
parser.add_argument('-desc', action='store_true', help='Compute image descriptors')
parser.add_argument('-bow', action='store_true', help='Compute Bag of Words')
parser.add_argument('-train', action='store_true', help='Train classifier')
parser.add_argument('-test', action='store_true', help='Test classifier')
parser.add_argument('-testone', action='store_true', help='Test classifier for one image independently')
args = parser.parse_args()
DESC = args.desc
BOW = args.bow
TRAIN = args.train
TEST = args.test
TESTONE = args.testone

#-----------------------------------------------------------------------------------------------------------
DESC = 'desc'
BOW = 'bow'
TRAIN = 'train'
TEST = 'test'
TESTONE = 'testone'
#-----------------------------------------------------------------------------------------------------------

IMG_PATH = 'images'
ANNOT_PATH = 'annotations'
DESC_PATH = 'descriptors.pkl'
BOW_PATH = 'bow.pkl'
SVM_TRAIN_PATH = 'model.pkl'
VARS_TRAIN_PATH = 'train.pkl'

db = Database(IMG_PATH, ANNOT_PATH)


labels_train = {
    'lower_bound': 0,
    'upper_bound': 10, #900
}

labels_test = {
    'lower_bound': 16,
    'upper_bound': 22, #900
}

keys = {
    'ISIC_0': 0,
    'ISIC_1': 1,
}

def main():
    global db

    with open("ISBI2016_ISIC_Part3_Training_GroundTruth.csv") as f:
        keys = {line.split(',')[0]: line.split(',')[1].partition("\n")[0] for line in f}

    descriptors = []
    features = []
    if not DESC:
        try:
            print('Loading descriptors file from path: ', BOW_PATH)
            descriptors = load_object(DESC_PATH)
            print('Finished loading descriptors')
        except Exception:
            print('Unable to load descriptors file')
            quit()

    print('Starting computing descriptors')
    file_it = labels_train['lower_bound']
    i = 0
    while file_it < labels_train['upper_bound'] :
        try:
            label = 'ISIC_'
            file_it = file_it + 1
            name = label + str(file_it).zfill(7)
            img = db.read_img(name)
            print('Computing descriptors for ' + str(label) + str(file_it).zfill(7) + ' of ' + str(labels_train['upper_bound']))
            img = gray(img)
            img = resize_img(img)
            feature = get_key_points(img)
            features.append((name, img, feature))
            if DESC:
                descriptors.extend(feature[1])
            i = i + 1
        except:
            continue
    labels_train['upper_bound'] = i

    file_it = labels_test['lower_bound']
    while file_it < labels_test['upper_bound'] :
        try:
            label = 'ISIC_'
            file_it = file_it + 1
            name = label + str(file_it).zfill(7)
            img = db.read_img(name)
            print('Computing descriptors for ' + str(label) + str(file_it).zfill(7) + ' of ' + str(labels_test['upper_bound']))
            img = gray(img)
            img = resize_img(img)
            feature = get_key_points(img)
            features.append((name, img, feature))
            if DESC:
                descriptors.extend(feature[1])
            i = i + 1
        except:
            continue
    labels_test['lower_bound'] = labels_train['upper_bound']
    labels_test['upper_bound'] = i


    print('Finished computing descriptors')
    print('Storing descriptors')
    store_object(descriptors, DESC_PATH)
    print('Finished storing descriptors')
    
    descriptors = array_to_np(descriptors)
    dictionary = None

    if not BOW:
        try:
            print('Loading dictionary file from path: ', BOW_PATH)
            dictionary = load_object(BOW_PATH)
            print('Finished loading dictionary')
        except:
            print('Unable load dictionary file')
            quit()
    else:
        print('Computing bag of words')
        bow = bow_trainer(100)
        dictionary = bow_cluster(bow, descriptors)
        print('Finished computing bag of words')
        print('Storing bag of words')
        store_object(dictionary, BOW_PATH)
        print('Finished storing bag of words')

    X = []
    y = []
    print('Computing target variables')
    extractor = bow_extractor(dictionary)
    for name, img, feature in features:
            X.append(bow_extract(extractor, img, feature[0])[0])
            y.append(1 if keys[name] == 'malignant' else 0)
    print('Finish computing target variables')

    svm = None
    if not TRAIN:
        try:
            print('Loading model file from path: ', SVM_TRAIN_PATH)
            svm = load_svm(SVM_TRAIN_PATH)
            print('Finished loading model')
        except:
            print('Cant load model file')
            quit()
    else:
        print('Training model')
        svm = create_svm()
        X_train = X[labels_train['lower_bound']:labels_train['upper_bound']]
        y_train = y[labels_train['lower_bound']:labels_train['upper_bound']]
        train_svm(svm, X_train, y_train)
        print('Finished training model')

        print('Saving model')
        store_object([X,y], VARS_TRAIN_PATH)
        store_svm(svm, SVM_TRAIN_PATH)
        print('Finished saving model')


    if TEST:
        print('Testing model')
        X_test = X[labels_test['lower_bound']:labels_test['upper_bound']]
        y_test = y[labels_test['lower_bound']:labels_test['upper_bound']]
        pred = np.squeeze(test_svm(svm, X_test)[1].astype(int))
        print('Finished testing model')

        print('Confusion Matrix: \n' + str(confusion_matrix(y_test,pred)))
        correct = 0
        for i in range(len(y_test)):
            if y_test[i] == pred[i]:
                correct += 1
        print('Accuracy: ', str(correct/len(y_test)*100), '%')

if __name__ == "__main__":
    main()
