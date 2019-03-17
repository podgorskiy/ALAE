import random
import pickle
import zipfile

import numpy as np

from scipy import misc
import tqdm

for i in range(1):
    with open('data_fold_%d.pkl' % i, 'rb') as pkl:
        data_train = pickle.load(pkl)

    with open('data_fold_%d_lod_%d.pkl' % (i, 3), 'wb') as pkl:
        pickle.dump(data_train, pkl)
            
    for j in range(3):
        data_train_down = []
        for l, image in tqdm.tqdm(data_train):
            image_down = misc.imresize(image, [image.shape[0] // 2, image.shape[1] // 2])
            data_train_down.append((l, image_down))

        with open('data_fold_%d_lod_%d.pkl' % (i, 3 - j - 1), 'wb') as pkl:
            pickle.dump(data_train_down, pkl)

        data_train = data_train_down
    
