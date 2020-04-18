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

import os
import sys
import argparse
import logging
import tensorflow as tf
from defaults import get_cfg_defaults
import tqdm
import dareblopy as db
from PIL import Image


def split_tfrecord(cfg, logger):
    tfrecord_path = cfg.DATASET.FFHQ_SOURCE

    ffhq_size = cfg.DATASET.SIZE

    part_size = ffhq_size // cfg.DATASET.PART_COUNT

    logger.info("Splitting into % size parts" % part_size)

    chunk_size = 1024

    # # Commented code is for saving out samples of bedroom dataset
    # with tf.Graph().as_default(), tf.Session() as sess:
    #     ds = tf.data.TFRecordDataset(tfrecord_path % 8)
    #     batch = ds.batch(256).make_one_shot_iterator().get_next()
    #
    #     features = {
    #         # 'shape': db.FixedLenFeature([3], db.int64),
    #         'data': db.FixedLenFeature([3, 256, 256], db.uint8)
    #     }
    #     parser = db.RecordParser(features, False)
    #     try:
    #         path = 'dataset_samples/bedroom256x256'
    #         os.makedirs(path, exist_ok=True)
    #         records = sess.run(batch)
    #         k = 0
    #         for record in records:
    #             im = parser.parse_single_example(record)[0]
    #             im = im.transpose((1, 2, 0))
    #             image = Image.fromarray(im)
    #             image.save(path + '/' + str(k) + ".png")
    #             k += 1
    #
    #     except tf.errors.OutOfRangeError:
    #         pass

    for i in range(0, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
        part_num = 0
        with tf.Graph().as_default(), tf.Session() as sess:
            ds = tf.data.TFRecordDataset(tfrecord_path % i)
            batch = ds.batch(chunk_size).make_one_shot_iterator().get_next()
            while True:
                try:
                    part_path = cfg.DATASET.PATH % (i, part_num)
                    os.makedirs(os.path.dirname(part_path), exist_ok=True)
                    k = 0
                    with tf.python_io.TFRecordWriter(part_path) as writer:
                        for k in tqdm.tqdm(range(part_size // chunk_size)):
                            records = sess.run(batch)
                            for record in records:
                                writer.write(record)
                    part_num += 1
                except tf.errors.OutOfRangeError:
                    break


def run():
    parser = argparse.ArgumentParser(description="ALAE. Split LSUN bedroom into parts")
    parser.add_argument(
        "--config-file",
        default="configs/bedroom.yaml",
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

    fh = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    split_tfrecord(cfg, logger)


if __name__ == '__main__':
    run()

