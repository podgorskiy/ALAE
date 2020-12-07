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


import dnnlib.tflib
import pickle
from net import *
from model import Model
from launcher import run
from dataloader import *

from checkpointer import Checkpointer

from dlutils.pytorch import count_parameters
from dlutils import download
from defaults import get_cfg_defaults
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import utils

dnnlib.tflib.init_tf()

download.from_google_drive('1CIDc9i070KQhHlkr4yIwoJC8xqrwjE0_', directory="metrics")


def downscale(images):
    if images.shape[2] > 256:
        factor = images.shape[2] // 256
        images = torch.reshape(images,
                               [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor,
                                factor])
        images = torch.mean(images, dim=(3, 5))
    images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)
    return images


class LPIPS:
    def __init__(self, cfg, num_images, minibatch_size):
        self.num_images = num_images
        self.minibatch_size = minibatch_size
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, encoder, lod):
        gpu_count = torch.cuda.device_count()
        distance_measure = pickle.load(open('metrics/vgg16_zhang_perceptual.pkl', 'rb'))

        dataset = TFRecordsDataset(self.cfg, logger, rank=0, world_size=1, buffer_size_mb=128,
                                   channels=self.cfg.MODEL.CHANNELS, train=False)

        dataset.reset(lod + 2, self.minibatch_size)
        batches = make_dataloader(self.cfg, logger, dataset, self.minibatch_size, 0,)

        distance = []
        num_images_processed = 0
        for idx, x in tqdm(enumerate(batches)):
            torch.cuda.set_device(0)
            x = (x / 127.5 - 1.)

            Z = encoder(x, lod, 1)
            Z = Z.repeat(1, mapping.num_layers, 1)

            images = decoder(Z, lod, 1.0, noise=True)

            images = downscale(images)
            images_ref = downscale(torch.tensor(x))

            res = distance_measure.run(images, images_ref, num_gpus=gpu_count, assume_frozen=True)
            distance.append(res)
            num_images_processed += x.shape[0]
            if num_images_processed > self.num_images:
                break

        print(len(distance))
        logger.info("Result = %f" % (np.asarray(distance).mean()))


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=None,
        truncation_cutoff=None,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder

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
        # 'encoder_s': encoder,
        'generator_s': decoder,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg_s': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Evaluating LPIPS metric")

    decoder = nn.DataParallel(decoder)
    encoder = nn.DataParallel(encoder)

    with torch.no_grad():
        ppl = LPIPS(cfg, num_images=10000, minibatch_size=16 * torch.cuda.device_count())
        ppl.evaluate(logger, mapping_fl, decoder, encoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-lpips', default_config='configs/experiment_celeba.yaml',
        world_size=gpu_count, write_log="metrics/lpips_score.txt")
