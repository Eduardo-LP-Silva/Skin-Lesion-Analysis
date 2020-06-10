from sklearn import svm
import cv2
import numpy as np
import glob
import pickle
import imutils



WIDTH = 512

def resize(img):
    return imutils.resize(img, width=WIDTH)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_np(arr):
    return np.array(arr)

def store(desc_list, path):
    pickle.dump(desc_list, open(path, 'wb'))

def load(path):
    return pickle.load(open(path, 'rb'))

def svm_store(svm, path):
    svm.save(path)

def svm_load(path):
    return cv2.ml.SVM_load(path)

class Database():
    def __init__(self, imgs):
        self.imgs = imgs

    def read_img(self, name):
        return cv2.imread(self._get_img_path(name))

    def _get_img_path(self, name):
        return glob.glob('./'+self.imgs + '/' + name + '.jpg', recursive=True)[0]
    

def calculate_key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return (sift.detectAndCompute(img, None))

def train_bow(dictionary_size=50):
    return cv2.BOWKMeansTrainer(dictionary_size)

def cluster_bow(trainer, desc):
    return trainer.cluster(desc)

def bow_retrieve(extractor, img, desc):
    return extractor.compute(img, desc)

def bow_retriever(dictionary):
    extractor = cv2.BOWImgDescriptorExtractor(cv2.xfeatures2d.SIFT_create(), cv2.FlannBasedMatcher())
    extractor.setVocabulary(dictionary)
    return extractor

def svm_create():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return svm

def svm_train(svm,X,y):
    svm.trainAuto(to_np(X), cv2.ml.ROW_SAMPLE, to_np(y), kFold=15)
    
def svm_test(svm, pred):
    return svm.predict(np.array(pred))



if __name__ == "__main__":
    pass