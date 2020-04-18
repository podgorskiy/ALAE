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

import torch.utils.data
from torchvision.utils import save_image
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from PIL import Image
import random


lreq.use_implicit_lreq.set(True)


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
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
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

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    path = cfg.DATASET.SAMPLES_PATH
    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    def do_attribute_traversal(path, attrib_idx, start, end):
        img = np.asarray(Image.open(path))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        factor = x.shape[2] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[2] == im_size
        _latents = encode(x[None, ...].cuda())
        latents = _latents[0, 0]

        latents -= model.dlatent_avg.buff.data[0]

        w0 = torch.tensor(np.load("principal_directions/direction_%d.npy" % attrib_idx), dtype=torch.float32)

        attr0 = (latents * w0).sum()

        latents = latents - attr0 * w0

        def update_image(w):
            with torch.no_grad():
                w = w + model.dlatent_avg.buff.data[0]
                w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)

                layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (7 + 1) * 2
                mixing_cutoff = cur_layers
                styles = torch.where(layer_idx < mixing_cutoff, w, _latents[0])

                x_rec = decode(styles)
                return x_rec

        traversal = []

        r = 7
        inc = (end - start) / (r - 1)

        for i in range(r):
            W = latents + w0 * (attr0 + start)
            im = update_image(W)

            traversal.append(im)
            attr0 += inc
        res = torch.cat(traversal)

        indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]
        labels = ["gender",
                  "smile",
                  "attractive",
                  "wavy-hair",
                  "young",
                  "big_lips",
                  "big_nose",
                  "chubby",
                  "glasses",
                  ]
        save_image(res * 0.5 + 0.5, "make_figures/output/%s/traversal_%s.jpg" % (
            cfg.NAME, labels[indices.index(attrib_idx)]), pad_value=1)

    do_attribute_traversal(path + '/00049.png', 0, 0.6, -34)
    do_attribute_traversal(path + '/00125.png', 1, -3, 15.0)
    do_attribute_traversal(path + '/00057.png', 3, -2, 30.0)
    do_attribute_traversal(path + '/00031.png', 4, -10, 30.0)
    do_attribute_traversal(path + '/00088.png', 10, -0.3, 30.0)
    do_attribute_traversal(path + '/00004.png', 11, -25, 20.0)
    do_attribute_traversal(path + '/00012.png', 17, -40, 40.0)
    do_attribute_traversal(path + '/00017.png', 19, 0, 30.0)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-traversals', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
