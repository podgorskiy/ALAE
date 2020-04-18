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

"""Create a tfrecords for celeba128x128 training. """

import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import random
import argparse
import os
import tensorflow as tf
import imageio
from PIL import Image


def prepare_celeba(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    with open("/data/datasets/CelebA/Eval/list_eval_partition.txt") as f:
        lineList = f.readlines()
    lineList = [x[:-1].split(' ') for x in lineList]

    split_map = {}
    for x in lineList:
        split_map[int(x[0][:-4])] = int(x[1])

    os.makedirs(directory, exist_ok=True)

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

    def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
        # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
        if crop_w is None:
            crop_w = crop_h # the width and height after cropped
        h, w = x.shape[:2]
        j = int(round((h - crop_h)/2.)) + 15
        i = int(round((w - crop_w)/2.))
        return np.array(Image.fromarray(x[j:j+crop_h, i:i+crop_w]).resize([resize_w, resize_w]))

    archive = zipfile.ZipFile(os.path.join(directory, '/data/datasets/CelebA/Img/img_align_celeba.zip'), 'r')

    names = archive.namelist()

    names = [x for x in names if x[-4:] == '.jpg']

    if train:
        names = [x for x in names if split_map[int(x[:-4][-6:])] != 2]
    else:
        names = [x for x in names if split_map[int(x[:-4][-6:])] == 2]

    count = len(names)
    print("Count: %d" % count)

    names = [x for x in names if x[-10:] not in corrupted]

    random.seed(0)
    random.shuffle(names)

    folds = cfg.DATASET.PART_COUNT
    celeba_folds = [[] for _ in range(folds)]

    spread_identiteis_across_folds = True

    if spread_identiteis_across_folds:
        # Reading indetities
        # Has format of
        # 000001.jpg 2880
        # 000002.jpg 2937
        with open("/data/datasets/CelebA/Anno/identity_CelebA.txt") as f:
            lineList = f.readlines()

        lineList = [x[:-1].split(' ') for x in lineList]

        identity_map = {}
        for x in lineList:
            identity_map[x[0]] = int(x[1])

        names = [(identity_map[x.split('/')[1]], x) for x in names]

        class_bins = {}

        for x in names:
            if x[0] not in class_bins:
                class_bins[x[0]] = []
            img_file_name = x[1]
            class_bins[x[0]].append((x[0], img_file_name))

        left_overs = []

        for _class, filenames in class_bins.items():
            count = len(filenames)
            print("Class %d count: %d" % (_class, count))

            count_per_fold = count // folds

            for i in range(folds):
                celeba_folds[i] += filenames[i * count_per_fold: (i + 1) * count_per_fold]

            left_overs += filenames[folds * count_per_fold:]

        leftover_per_fold = len(left_overs) // folds
        for i in range(folds):
            celeba_folds[i] += left_overs[i * leftover_per_fold: (i + 1) * leftover_per_fold]

        for i in range(folds):
            random.shuffle(celeba_folds[i])

        # strip ids
        for i in range(folds):
            celeba_folds[i] = [x[1] for x in celeba_folds[i]]

        print("Folds sizes:")
        for i in range(len(celeba_folds)):
            print(len(celeba_folds[i]))
    else:
        count_per_fold = count // folds
        for i in range(folds):
            celeba_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(folds):
        images = []
        for x in tqdm.tqdm(celeba_folds[i]):
            imgfile = archive.open(x)
            image = center_crop(imageio.imread(imgfile.read()))
            images.append((int(x[:-4][-6:]), image.transpose((2, 0, 1))))

        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

        if train:
            part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        else:
            part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)

        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

        random.shuffle(images)

        for label, image in images:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        for j in range(5):
            images_down = []

            for label, image in tqdm.tqdm(images):
                h = image.shape[1]
                w = image.shape[2]
                image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)

                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)

                image_down = image_down.view(3, h // 2, w // 2).numpy()
                images_down.append((label, image_down))

            if train:
                part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)
            else:
                part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)

            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            for label, image in images_down:
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()

            images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. Prepare tfrecords for celeba128x128")
    parser.add_argument(
        "--config-file",
        default="configs/celeba.yaml",
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

    prepare_celeba(cfg, logger, True)
    prepare_celeba(cfg, logger, False)


if __name__ == '__main__':
    run()

