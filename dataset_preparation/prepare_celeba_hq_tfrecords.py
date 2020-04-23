import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging

from dlutils import download

from scipy import misc
from net import *
import numpy as np
import pickle
import random
import argparse
import os
from dlutils.pytorch.cuda_helper import *
import tensorflow as tf
import imageio
from PIL import Image


def prepare_celeba(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    os.makedirs(directory, exist_ok=True)

    images = []
    # The official way of generating CelebA-HQ can be challenging.
    # Please refer to this page: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download
    # You can get pre-generated dataset from: https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P
    source_path = '/data/datasets/celeba-hq/data1024x1024'
    for filename in tqdm.tqdm(os.listdir(source_path)):
        images.append((int(filename[:-4]), filename))

    print("Total count: %d" % len(images))
    if train:
        images = images[:cfg.DATASET.SIZE]
    else:
        images = images[cfg.DATASET.SIZE_TEST:]

    count = len(images)
    print("Count: %d" % count)

    random.seed(0)
    random.shuffle(images)

    folds = cfg.DATASET.PART_COUNT
    celeba_folds = [[] for _ in range(folds)]

    count_per_fold = count // folds
    for i in range(folds):
        celeba_folds[i] += images[i * count_per_fold: (i + 1) * count_per_fold]

    for i in range(folds):
        if train:
            path = cfg.DATASET.PATH
        else:
            path = cfg.DATASET.PATH_TEST

        writers = {}
        for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            part_path = path % (lod, i)
            os.makedirs(os.path.dirname(part_path), exist_ok=True)
            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            writers[lod] = tfr_writer

        for label, filename in tqdm.tqdm(celeba_folds[i]):
            img = np.asarray(Image.open(os.path.join(source_path, filename)))
            img = img.transpose((2, 0, 1))
            for lod in range(cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, -1):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()]))}))
                writers[lod].write(ex.SerializeToString())

                image = torch.tensor(np.asarray(img, dtype=np.float32)).view(1, 3, img.shape[1], img.shape[2])
                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8).view(3, image.shape[2] // 2, image.shape[3] // 2).numpy()

                img = image_down


def run():
    parser = argparse.ArgumentParser(description="Adversarial, hierarchical style VAE")
    parser.add_argument(
        "--config-file",
        default="configs/celeba-hq256.yaml",
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

