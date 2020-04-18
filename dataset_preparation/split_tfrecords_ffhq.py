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


def split_tfrecord(cfg, logger):
    tfrecord_path = cfg.DATASET.FFHQ_SOURCE

    ffhq_train_size = 60000

    part_size = ffhq_train_size // cfg.DATASET.PART_COUNT

    logger.info("Splitting into % size parts" % part_size)

    for i in range(2, cfg.DATASET.MAX_RESOLUTION_LEVEL + 1):
        with tf.Graph().as_default(), tf.Session() as sess:
            ds = tf.data.TFRecordDataset(tfrecord_path % i)
            ds = ds.batch(part_size)
            batch = ds.make_one_shot_iterator().get_next()
            part_num = 0
            while True:
                try:
                    records = sess.run(batch)
                    if part_num < cfg.DATASET.PART_COUNT:
                        part_path = cfg.DATASET.PATH % (i, part_num)
                        os.makedirs(os.path.dirname(part_path), exist_ok=True)
                        with tf.python_io.TFRecordWriter(part_path) as writer:
                            for record in records:
                                writer.write(record)
                    else:
                        part_path = cfg.DATASET.PATH_TEST % (i, part_num - cfg.DATASET.PART_COUNT)
                        os.makedirs(os.path.dirname(part_path), exist_ok=True)
                        with tf.python_io.TFRecordWriter(part_path) as writer:
                            for record in records:
                                writer.write(record)
                    part_num += 1
                except tf.errors.OutOfRangeError:
                    break


def run():
    parser = argparse.ArgumentParser(description="ALAE. Split FFHQ into parts for training and testing")
    parser.add_argument(
        "--config-file",
        default="configs/ffhq.yaml",
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

