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

from dataloader import *
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from model import Model
from tqdm import tqdm
from launcher import run
from net import *
import numpy as np
import tensorflow as tf


class ImageGenerator:
    def __init__(self, cfg, num_samples, minibatch_gpu):
        self.num_samples = num_samples
        self.minibatch_size = minibatch_gpu
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, lod):
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        tfr_writer = tf.python_io.TFRecordWriter("principal_directions/generated_data.000", tfr_opt)

        rnd = np.random.RandomState(5)

        for _ in tqdm(range(0, self.num_samples, self.minibatch_size)):
            torch.cuda.set_device(0)
            latents = rnd.randn(self.minibatch_size, self.cfg.MODEL.LATENT_SPACE_SIZE)
            lat = torch.tensor(latents).float().cuda()

            dlat = mapping(lat)
            images = decoder(dlat, lod, 1.0, True)

            # Downsample to 256x256. The attribute classifiers were built for 256x256.
            factor = images.shape[2] // 256
            if factor != 1:
                images = torch.nn.functional.avg_pool2d(images, factor, factor)
            images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)

            for i, img in enumerate(images):
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                    'lat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lat[i].cpu().numpy().tostring()])),
                    'dlat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dlat[i, 0].cpu().numpy().tostring()]))}))
                tfr_writer.write(ex.SerializeToString())


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda()
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Generating...")

    decoder = nn.DataParallel(decoder)
    mapping_fl = nn.DataParallel(mapping_fl)

    with torch.no_grad():
        gen = ImageGenerator(cfg, num_samples=60000, minibatch_gpu=8)
        gen.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-generate-images-for-attribute-classifications',
        default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
