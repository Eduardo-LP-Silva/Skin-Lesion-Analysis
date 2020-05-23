from sklearn import svm
import xml.etree.ElementTree as ET
import pickle
import imutils
import cv2
import numpy as np
import glob
import os


WIDTH = 512

class Database():
    def __init__(self, path_imgs, path_annot):
        self.path_imgs = path_imgs
        self.path_annot = path_annot

    def read_img(self, name):
        return cv2.imread(self._get_img_path(name))

    def _get_annot_path(self, name):
        return glob.glob(self.path_annot + '/' + name + '.xml', recursive=True)[0]

    def _get_img_path(self, name):
        return glob.glob('./'+self.path_imgs + '/' + name + '.jpg', recursive=True)[0]
    

def get_key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return (sift.detectAndCompute(img, None))

def bow_trainer(dictionary_size=50):
    return cv2.BOWKMeansTrainer(dictionary_size)

def bow_cluster(trainer, desc):
    return trainer.cluster(desc)

def bow_extractor(dictionary):
    matcher = cv2.FlannBasedMatcher()
    detector = cv2.xfeatures2d.SIFT_create()
    extractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
    extractor.setVocabulary(dictionary)
    return extractor

def bow_extract(extractor, img, desc):
    return extractor.compute(img, desc)

def create_svm():
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.setType(cv2.ml.SVM_C_SVC)
    return svm

def train_svm(svm,X,y):
    svm.trainAuto(array_to_np(X), cv2.ml.ROW_SAMPLE, array_to_np(y), kFold=15)
    
def test_svm(svm, pred):
    return svm.predict(np.array(pred))

def resize_img(img):
    return imutils.resize(img, width=WIDTH)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def array_to_np(arr):
    return np.array(arr)

def store_object(desc_list, path):
    pickle.dump(desc_list, open(path, 'wb'))

def load_object(path):
    return pickle.load(open(path, 'rb'))

def store_svm(svm, path):
    svm.save(path)

def load_svm(path):
    return cv2.ml.SVM_load(path)


if __name__ == "__main__":
    pass