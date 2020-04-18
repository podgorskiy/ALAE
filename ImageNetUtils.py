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

"""Create a tfrecords for ImageNET. """

import os
import scipy.io as sio
import torch
from PIL import Image
import random
import argparse
from defaults import get_cfg_defaults
import sys
import logging

import tensorflow as tf
from torchvision.transforms import functional as F
from torch.nn.functional import avg_pool2d
from utils import cache
import numpy as np
import tqdm
from multiprocessing import Pool
from threading import Thread


def process_fold(i, path, image_folds, train_root, wnid_to_indx, fixed=False):
    writers = {}
    for lod in range(8, 1, -1):
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        part_path = path % (lod, i)
        os.makedirs(os.path.dirname(part_path), exist_ok=True)
        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
        writers[lod] = tfr_writer

    for s, image in image_folds[i]:
        im = os.path.join(train_root, s, image)
        img = Image.open(im)
        if fixed:
            img = F.resize(img, 288)
            img = F.center_crop(img, 256)
        else:
            img = F.resize(img, 288)
            img = F.center_crop(img, 288)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], (1, 1, 3))
        img = img.transpose((2, 0, 1))
        if img.shape[0] > 3:
            img = img[:3]

        for lod in range(8, 1, -1):
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[wnid_to_indx[s]])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))}))
            writers[lod].write(ex.SerializeToString())

            image = torch.tensor(np.asarray(img, dtype=np.float32)).view(1, 3, img.shape[1], img.shape[2])
            image_down = avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8).view(3, image.shape[2] // 2,
                                                                                            image.shape[3] // 2).numpy()

            img = image_down

    for lod in range(8, 1, -1):
        writers[lod].close()


def parse_meta_mat(devkit_root):
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def parse_val_groundtruth_txt(devkit_root):
    file = os.path.join(devkit_root, "data",
                        "ILSVRC2012_validation_ground_truth.txt")
    with open(file, 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) for val_idx in val_idcs]


@cache
def get_names(train_root):
    names = []
    sets = os.listdir(train_root)
    for s in sets:
        images = os.listdir(os.path.join(train_root, s))
        names += [(s, im) for im in images]
    return names


def prepare_imagenet(cfg, logger):
    devkit_root = "/data/datasets/ImageNet_bak/ILSVRC2012_devkit_t12"
    idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
    val_idcs = parse_val_groundtruth_txt(devkit_root)
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

    for i in range(1, 1001):
        w = idx_to_wnid[i]
        c = wnid_to_classes[w]
        print("%d - %s" % (i, c))

    wnid_to_indx = dict([(v, k - 1) for k, v in idx_to_wnid.items()])

    torch.save((wnid_to_classes, val_wnids), os.path.join("", "meta"))

    train_root = "/data/datasets/ImageNet_bak/raw-data/train"
    validation_root = "/data/datasets/ImageNet_bak/raw-data/validation"

    ###
    logger.info("Savingexamples")

    path = 'dataset_samples/imagenet256x256'
    os.makedirs(path, exist_ok=True)
    k = 0
    names = get_names(train_root)
    random.shuffle(names)
    for s, image in names:
        im = os.path.join(train_root, s, image)
        img = Image.open(im)
        img = F.resize(img, 288)
        img = F.center_crop(img, 256)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], (1, 1, 3))
        img = img.transpose((2, 0, 1))
        if img.shape[0] > 3:
            img = img[:3]
        img = img.transpose((1, 2, 0))
        img = Image.fromarray(img)
        img.save(path + '/' + str(k) + ".png")
        k += 1
        if k == 2000:
            break
    ###
    exit()

    if True:
        random.seed(0)

        names = get_names(train_root)
        random.shuffle(names)

        folds = 16 # cfg.DATASET.PART_COUNT
        image_folds = [[] for _ in range(folds)]

        count_per_fold = len(names) // folds
        for i in range(folds):
            image_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]

        threads = []
        for i in range(folds):
            thread = Thread(target=process_fold, args=(i, cfg.DATASET.PATH, image_folds, train_root, wnid_to_indx, False))
            thread.start()
            threads.append(thread)

        for i in range(folds):
            threads[i].join()
    if False:
        random.seed(0)

        names = get_names(validation_root)
        random.shuffle(names)

        folds = 1 # cfg.DATASET.PART_COUNT
        image_folds = [[] for _ in range(folds)]

        count_per_fold = len(names) // folds
        for i in range(folds):
            image_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]

        threads = []
        for i in range(folds):
            thread = Thread(target=process_fold, args=(i, cfg.DATASET.PATH_TEST, image_folds, validation_root, wnid_to_indx, True))
            thread.start()
            threads.append(thread)

        for i in range(folds):
            threads[i].join()

    print(idx_to_wnid, wnid_to_classes)


def run():
    parser = argparse.ArgumentParser(description="ALAE imagenet")
    parser.add_argument(
        "--config-file",
        default="configs/imagenet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_imagenet(cfg, logger)


if __name__ == '__main__':
    run()

