import tensorflow as tf
import IPython.display as display
import numpy as np
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

print(tf.__version__)

data_dir = pathlib.Path('./resources/Task1/TrainingSorted')
image_count = len(list(data_dir.glob('*/*.jpg')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

print(image_count)
print(CLASS_NAMES)