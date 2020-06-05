from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from imutils import paths
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import tensorflow as tf
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_DIR = '/content/drive/My Drive/Colab Notebooks/data/task1/training_sorted_resized'
TEST_DATA_DIR = '../data/task1/testing_sorted'
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 15
EPOCHS = 20

def load_dataset(dir):
    print("[INFO] Loading images")
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.3)
    train_gen = image_generator.flow_from_directory(directory=dir, color_mode='rgb', batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', subset='training')
    valid_gen = image_generator.flow_from_directory(directory=dir, color_mode='rgb', batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', subset='validation')

    return train_gen, valid_gen


# plots images with labels within jupyter notebook
def plots(ims, figsize=(24,12), rows=4, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=32)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def build_model():
    print("[INFO] Building model")
    input_layer = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    input_tensor = tf.keras.applications.vgg16.preprocess_input(input_layer)
    print(input_layer)
    print(input_tensor)
    baseModel = VGG16(weights='imagenet', include_top=True, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), input_tensor=input_tensor)

    model = tf.keras.Sequential()

    for layer in baseModel.layers[:-1]:
      layer.trainable = False #Better results when vgg layers are frozen (only in imagenet)
      model.add(layer)
    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    return model

def train_model(model, train_gen, val_gen):
    print("[INFO] Training model")
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    H = model.fit(x=train_gen, validation_data=val_gen, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    return model, H

train_gen, valid_gen = load_dataset(TRAIN_DATA_DIR)

'''
imgs, labels = next(train_gen)
plots(imgs, titles=labels)
'''

model = build_model()
model, history = train_model(model, train_gen, valid_gen)
name = 'model_BS' + str(BATCH_SIZE) + '_E'+ str(EPOCHS)
model.save('/content/drive/My Drive/Colab Notebooks/' + name)

plt.plot(np.arange(0, 30), history.history["loss"], label="loss")
plt.plot(np.arange(0, 30), history.history["accuracy"], label="accuracy")
#plt.plot(np.arange(0, 30), history.history["precision"], label="precision")
plt.plot(np.arange(0, 30), history.history["recall"], label="recall")
plt.title("Training Metrics")
plt.xlabel("Epoch #")
plt.ylabel("Metrics")
plt.legend(loc="best")
plt.savefig('/content/drive/My Drive/Colab Notebooks/' + name + '/plot')