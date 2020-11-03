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
import os
from PIL import Image


lreq.use_implicit_lreq.set(True)


def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder=cfg.MODEL.MAPPING_FROM_ECOG,
        z_regression=cfg.MODEL.Z_REGRESSION,
        average_w = cfg.MODEL.AVERAGE_W,
        temporal_w = cfg.MODEL.TEMPORAL_W,
        global_w = cfg.MODEL.GLOBAL_W,
        temporal_global_cat = cfg.MODEL.TEMPORAL_GLOBAL_CAT,
        spec_chans = cfg.DATASET.SPEC_CHANS,
        temporal_samples = cfg.DATASET.TEMPORAL_SAMPLES,
        init_zeros = cfg.MODEL.TEMPORAL_W,
        residual = cfg.MODEL.RESIDUAL,
        w_classifier = cfg.MODEL.W_CLASSIFIER,
        uniq_words = cfg.MODEL.UNIQ_WORDS,
        attention = cfg.MODEL.ATTENTION,
        cycle = cfg.MODEL.CYCLE,
        w_weight = cfg.TRAIN.W_WEIGHT,
        cycle_weight=cfg.TRAIN.CYCLE_WEIGHT,
        attentional_style=cfg.MODEL.ATTENTIONAL_STYLE,
        heads = cfg.MODEL.HEADS,
        suploss_on_ecog = cfg.MODEL.SUPLOSS_ON_ECOGF,
        less_temporal_feature = cfg.MODEL.LESS_TEMPORAL_FEATURE,
        ppl_weight=cfg.MODEL.PPL_WEIGHT,
        ppl_global_weight=cfg.MODEL.PPL_GLOBAL_WEIGHT,
        ppld_weight=cfg.MODEL.PPLD_WEIGHT,
        ppld_global_weight=cfg.MODEL.PPLD_GLOBAL_WEIGHT,
        common_z = cfg.MODEL.COMMON_Z,
    )
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

    extra_checkpoint_data = checkpointer.load(file_name='./training_artifacts/ecog_residual_latent128_temporal_lesstemporalfeature_noprogressive_HBw_ppl_ppld_localreg_debug/model_tmp_lod6.pth')
    # extra_checkpoint_data = checkpointer.load(file_name='./training_artifacts/ecog_residual_cycle_attention3264wIN_specchan64_more_attentfeatures/model_tmp_lod4.pth')

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT
    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
            Z = (Z[0].repeat(1, model.mapping_fl.num_layers, 1, 1),Z[1].repeat(1, model.mapping_fl.num_layers, 1))
        else:
            if cfg.MODEL.TEMPORAL_W:
                Z = Z.repeat(1, model.mapping_fl.num_layers, 1, 1)
            else:
                Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        # layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        # ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        # coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)
    rnd = np.random.RandomState(4)
    latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)

    path = cfg.DATASET.SAMPLES_PATH
    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    # pathA = 'kite.npy'
    # pathB = 'cat.npy'
    # pathC = 'hat.npy'
    # pathD = 'cake.npy'
    pathA = 'vase.npy'
    pathB = 'cow.npy'
    pathC = 'hat.npy'
    pathD = 'cake.npy'


    # def open_image(filename):
    #     img = np.asarray(Image.open(path + '/' + filename))
    #     if img.shape[2] == 4:
    #         img = img[:, :, :3]
    #     im = img.transpose((2, 0, 1))
    #     x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
    #     if x.shape[0] == 4:
    #         x = x[:3]
    #     factor = x.shape[2] // im_size
    #     if factor != 1:
    #         x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
    #     assert x.shape[2] == im_size
    #     _latents = encode(x[None, ...].cuda())
    #     latents = _latents[0, 0]
    #     return latents
    def open_image(filename):
        im = np.load(os.path.join(path, filename))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda()
        factor = x.shape[1] // im_size
        if factor != 1:
            x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
        assert x.shape[1] == im_size
        _latents = encode(x[None, ...].cuda())
        if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
            latents = (_latents[0][0,0],_latents[1][0,0])
        else:
            latents = _latents[0, 0]
        return latents

    def make(w):
        with torch.no_grad():
            if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
                w = (w[0][None, None, ...].repeat(1, model.mapping_fl.num_layers, 1, 1),w[1][None, None, ...].repeat(1, model.mapping_fl.num_layers, 1))
            else:
                if cfg.MODEL.TEMPORAL_W:
                    w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1, 1)
                else:
                    w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)
            x_rec = decode(w)
            return x_rec

    wa = open_image(pathA)
    wb = open_image(pathB)
    wc = open_image(pathC)
    wd = open_image(pathD)
    import pdb;pdb.set_trace()
    height = 10
    width = 10

    images = []

    for i in range(height):
        for j in range(width):
            kv = i / (height - 1.0)
            kh = j / (width - 1.0)

            ka = (1.0 - kh) * (1.0 - kv)
            kb = kh * (1.0 - kv)
            kc = (1.0 - kh) * kv
            kd = kh * kv

            if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
                w = ((1-kh) * wa[0] + kh * wb[0] , (1-kv) * wa[1] + kv * wb[1])
            else:
                w = ka * wa + kb * wb + kc * wc + kd * wd

            interpolated = make(w)
            images.append(interpolated)

    images = torch.cat(images)
    images = images.permute(0,1,3,2)
    os.makedirs('make_figures/output/%s' % cfg.NAME, exist_ok=True)
    save_image(images * 0.5 + 0.5, 'make_figures/output/%s/interpolations_vase_cow.png' % cfg.NAME, nrow=width)
    save_image(images * 0.5 + 0.5, 'make_figures/output/%s/interpolations_vase_cow.jpg' % cfg.NAME, nrow=width)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-interpolations', default_config='configs/ecog_style2.yaml',
        world_size=gpu_count, write_log=False)
