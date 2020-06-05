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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support
from mlxtend.plotting import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
import pathlib
import math
import numpy as np
import tensorflow as tf
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DATA_DIR = '/content/drive/My Drive/Colab Notebooks/data/task1/training_sorted_resized'
TEST_DATA_DIR = '/content/drive/My Drive/Colab Notebooks/data/task1/test_sorted_resized'
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 15
EPOCHS = 5

def load_dataset(train=True):
    print("[INFO] Loading images")
    
    if train:
      image_generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.1)
      train_gen = image_generator.flow_from_directory(directory=TRAIN_DATA_DIR, color_mode='rgb', batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', subset='training')
      valid_gen = image_generator.flow_from_directory(directory=TRAIN_DATA_DIR, color_mode='rgb', batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', subset='validation')
      return train_gen, valid_gen
    else:
      image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
      test_gen = image_generator.flow_from_directory(directory=TEST_DATA_DIR, color_mode='rgb', batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical')

      return test_gen


# plots images with labels
def plots(ims, figsize=(24,12), rows=4, interp=False, titles=None, predictions=None, keys=None):
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
            title = ''
            prediction = ''

            if(titles is not None):
              if(keys is not None):
                title = keys[titles[i]]
              else:
                title = titles[i]

            title = 'Value: ' + str(title)
              
            if predictions is not None:
              if keys is not None:
                prediction = keys[predictions[i]]
              else:
                prediction = predictions[i]

              title = title + ' | Prediction: ' + str(prediction)

            sp.set_title(title, fontsize=18)
            sp.set
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def build_model():
    print("[INFO] Building model")
    input_layer = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    input_tensor = tf.keras.applications.vgg16.preprocess_input(input_layer)
    baseModel = VGG16(weights='imagenet', include_top=True, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), input_tensor=input_tensor)

    model = tf.keras.Sequential()

    for layer in baseModel.layers[:-1]:
      #layer.trainable = False
      model.add(layer)

    '''
    layer_no = len(model.layers) - 1

    for i in range(layer_no, layer_no - 4, -1):
      model.layers[i].trainable = True
    '''
    
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    return model

def compile_model(model):
  model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

def train_model(model, train_gen, val_gen):
    print("[INFO] Training model")
    compile_model(model)

    H = model.fit(x=train_gen, validation_data=val_gen, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    return model, H

def test_model(model, test_gen, compile=False):
  print("[INFO] Testing model")

  if compile:
    compile_model(model)

  predictions = model.predict(x=test_gen, batch_size=BATCH_SIZE, verbose=1)

  return predictions

def get_generator_data(gen, batch_size):
  examples = len(gen.filenames)
  gen_calls = int(math.ceil(examples / (1.0 * batch_size)))

  images = []
  labels = []

  for i in range(0, gen_calls):
    images.extend(np.array(gen[i][0]))
    labels.extend(np.array(gen[i][1]))
    '''
    batch_labels = np.array(gen[i][1])
    for label in batch_labels:
      labels.extend(label)
    '''
    

  return images, labels

train_gen, valid_gen = load_dataset(True)

imgs, labels = next(train_gen)
plots(imgs, titles=np.argmax(labels, axis=1), keys=list(train_gen.class_indices.keys()))

model = build_model()
model, history = train_model(model, train_gen, valid_gen)
name = 'model_BS' + str(BATCH_SIZE) + '_E'+ str(EPOCHS)
#model.save('/content/drive/My Drive/Colab Notebooks/' + name)

plt.plot(np.arange(0, EPOCHS), history.history["loss"], label="loss")
plt.plot(np.arange(0, EPOCHS), history.history["accuracy"], label="accuracy", )
#plt.plot(np.arange(0, 30), history.history["precision"], label="precision")
#plt.plot(np.arange(0, EPOCHS), history.history["recall"], label="recall")
plt.ylim(0, 1)
plt.title("Training Metrics")
plt.xlabel("Epoch #")
plt.ylabel("Metrics")
plt.legend(loc="best")
#plt.savefig('/content/drive/My Drive/Colab Notebooks/' + name + '/plot')

test_gen = load_dataset(False)

test_images, test_labels = get_generator_data(test_gen, BATCH_SIZE)
print(test_labels)

predictions = test_model(model, test_gen)

round_predictions = np.argmax(predictions, axis=1)
round_labels = np.argmax(test_labels, axis=1)
print(round_predictions)
print(round_labels)

'''
confusion_mtx = confusion_matrix(round_labels, round_predictions)
cm_plot_labels = test_gen.class_indices.keys()
plot_confusion_matrix(confusion_matrix, classes=range(2))
'''

plots(test_images[0:9], titles=round_labels, predictions=round_predictions, keys=list(train_gen.class_indices.keys()))

print("Accuracy: ", accuracy_score(round_labels, round_predictions))
print("Precision: ", precision_score(round_labels, round_predictions, average="micro"))
print('Recall (Sensitivitiy): ', recall_score(round_labels, round_predictions, average="macro"))
print('F1 score: ', f1_score(round_labels, round_predictions, average="weighted"))