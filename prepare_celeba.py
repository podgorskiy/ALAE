import random
import pickle
import zipfile

import numpy as np

from scipy import misc
import tqdm

from dlutils import download


download.from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM")


def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w]) 


archive = zipfile.ZipFile('img_align_celeba.zip', 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.jpg']

folds = 5

random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)

count_per_fold = count // folds

i = 0
im = 0
for x in tqdm.tqdm(names):
    imgfile = archive.open(x)
    image = center_crop(misc.imread(imgfile))
    images[x] = image
    im += 1

    if im == count_per_fold:
        output = open('data_fold_%d.pkl' % i, 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
