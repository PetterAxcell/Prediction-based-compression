import os
import tensorflow as tf
import fileio
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import itertools
import copy as cp
import random
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras import initializers
from contextlib import redirect_stdout

# ------------------ IMAGES ------------------ 
training_images = glob.glob("*224x512x680.raw")
img = fileio.loadImage(training_images[0])[0].astype('uint16')

img = np.array(img)
img = np.clip(img, 0, 6000)

mean_value = np.mean(img)
print(mean_value)

img  = img - mean_value
img = img.reshape(-1)

#== HISTOGRAM ==
array = np.array(img)
xmin = array.min()
xmax = 3000
range = xmax - xmin + 1
histogram = np.histogram(array)

ymin = histogram[0].min()
ymax = histogram[0].max()

print(histogram[0])

#== PLOT ==
fig, ax = plt.subplots(figsize=(5,2.7), layout='constrained')
n, bins, patches = ax.hist(array, facecolor='blue')
ax.set_xlabel('Value of pixel')
ax.set_ylabel('Number of pixels')
ax.set_title('Histogram of values to predict mean subtracted')
ax.axis(([xmin, xmax, ymin, ymax]))
ax.grid(True)
fig.savefig('mean_histogram_clipped_5000.png')