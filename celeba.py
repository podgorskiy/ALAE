import random
import pickle
import zipfile

from scipy import misc
import numpy as np
import tqdm

def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w]) 


with open("identity_CelebA.txt") as f:
  lineList = f.readlines()
  
lineList = [x[:-1].split(' ') for x in lineList]

identity_map = {}
for x in lineList:
    identity_map[x[0]] = int(x[1])

    
archive = zipfile.ZipFile('img_align_celeba.zip', 'r')

names = archive.namelist()

names = [(identity_map[x.split('/')[1]], x) for x in names if x[-4:]=='.jpg']

folds = 5

random.shuffle(names)

class_bins = {}

for x in tqdm.tqdm(names):
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    imgfile = archive.open(x[1])
    image = center_crop(misc.imread(imgfile))
    class_bins[x[0]].append((x[0], image))

celeba_folds = [[] for _ in range(folds)]

for _class, data in class_bins.items():
    count = len(data)
    print("Class %d count: %d" % (_class, count))

    count_per_fold = count // folds

    for i in range(folds):
        celeba_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]


print("Folds sizes:")
for i in range(len(celeba_folds)):
    print(len(celeba_folds[i]))

    output = open('data_fold_%d.pkl' % i, 'wb')
    pickle.dump(celeba_folds[i], output)
    output.close()
