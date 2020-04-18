# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Create a pickle for celeba128x128 training.  Not used in current pipeline"""

import random
import pickle
import zipfile

import numpy as np

from scipy import misc
import tqdm

from dlutils import download


corrupted = [
    '195995.jpg',
    '131065.jpg',
    '118355.jpg',
    '080480.jpg',
    '039459.jpg',
    '153323.jpg',
    '011793.jpg',
    '156817.jpg',
    '121050.jpg',
    '198603.jpg',
    '041897.jpg',
    '131899.jpg',
    '048286.jpg',
    '179577.jpg',
    '024184.jpg',
    '016530.jpg',
]

download.from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM")


def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.)) + 15
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w]) 


archive = zipfile.ZipFile('img_align_celeba.zip', 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.jpg']

count = len(names)
print("Count: %d" % count)

names = [x for x in names if x[-10:] not in corrupted]

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
