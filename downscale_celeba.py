import random
import pickle
import zipfile
import os 

import numpy as np

from scipy import misc
import tqdm

for i in range(5):
    with open('data_fold_%d.pkl' % i, 'rb') as pkl:
        data_train = pickle.load(pkl)
    os.rename('data_fold_%d.pkl' % i, 'data_fold_%d_lod_5.pkl' % i) 
        
    for j in range(5):
        data_train_down = []
        for image in tqdm.tqdm(data_train):
            image_down = misc.imresize(image, [image.shape[0] // 2, image.shape[1] // 2])
            data_train_down.append(image_down)

        with open('data_fold_%d_lod_%d.pkl' % (i, 5 - j - 1), 'wb') as pkl:
            pickle.dump(data_train_down, pkl)

        data_train = data_train_down
    
