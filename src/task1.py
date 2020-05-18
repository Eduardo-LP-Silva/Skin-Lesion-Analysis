import tensorflow as tf
import numpy as np
import os
import pathlib
# import IPython.display as display
# from PIL import Image
# import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE
DIRECTORY = '../data/task1/training_sorted'
IMG_WIDTH = 1022
IMG_HEIGHT = 767

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def load_dataset(data_dir):
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

def main() :
    data_dir = pathlib.Path(DIRECTORY)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    global CLASS_NAMES 
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    load_dataset(data_dir)

if __name__ == "__main__":
    main()