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
BATCH_SIZE = 30

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
    return tf.image.resize(img, np.array([IMG_HEIGHT, IMG_WIDTH]))

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def load_dataset(data_dir):
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # for image, label in labeled_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Label: ", label.numpy())
    return labeled_ds

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def train_VGG(dataset):
    input_tensor = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), batch_size=30, dtype=tf.float32) # batch_size=30
    input_tensor = tf.keras.applications.vgg16.preprocess_input(input_tensor)
    #input_tensor = tf.reshape(input_tensor, [IMG_HEIGHT, IMG_WIDTH, 3])
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='max')
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    #flat = tf.keras.layers.Flatten(data_format='channels_last')(model.outputs[0])
    #dense = tf.keras.layers.Dense(1024, activation='relu')(flat)
    #output = tf.keras.layers.Dense(2, )
    model.summary()
    history = model.fit(prepare_for_training(dataset), batch_size=30, epochs=2, verbose=1, steps_per_epoch=1000)

def main() :
    data_dir = pathlib.Path(DIRECTORY)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    global CLASS_NAMES 
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    dataset = load_dataset(data_dir)
    train_VGG(dataset)


if __name__ == "__main__":
    main()