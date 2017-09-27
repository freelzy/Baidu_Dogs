from keras.preprocessing.image import *
import h5py
import math
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
import os,sys
import shutil


gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
)

destdir = 'bddogtrain\\preview'
for i in range(100):
    print(i)
    rootdir ='bddogtrain\\train\\'+str(i)
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            img = load_img(rootdir+'\\'+filename)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            n = 0
            for batch in gen.flow(x, batch_size=10,save_to_dir=destdir, save_prefix=filename.split('.')[0], save_format='jpeg'):
                n += 1
                if n >= 1:
                    break

gc.collect()