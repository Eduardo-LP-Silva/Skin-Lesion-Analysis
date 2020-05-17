import tensorflow as tf
import IPython.display as display
import numpy as np
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = pathlib.Path('../data/task1/training_sorted')
image_count = len(list(data_dir.glob('*/*.jpg')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

print(image_count)
print(CLASS_NAMES)