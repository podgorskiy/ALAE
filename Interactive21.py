# Copyright 2019 Stanislav Pidhorskyi
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

from __future__ import print_function
import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import bimpy
import math

im_size = 128


def process_batch(batch):
    data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size: (y + 1) * im_size, x * im_size: (x + 1) * im_size] = image * 0.5 + 0.5


def main(model_filename_enc, model_filename_dec):
    layer_count = 6
    latent_size = 256
    encoder = Encoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    decoder = Decoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    encoder.cuda()
    decoder.cuda()
    try:
        encoder.load_state_dict(torch.load(model_filename_enc))
        decoder.load_state_dict(torch.load(model_filename_dec))
    except RuntimeError:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        encoder.load_state_dict(torch.load(model_filename_enc))
        decoder.load_state_dict(torch.load(model_filename_dec))
        encoder = encoder.module
        decoder = decoder.module
    encoder.eval()
    decoder.eval()

    ids = list(range(36))

    print("Trainable parameters:")
    count_parameters(encoder)
    count_parameters(decoder)

    with open('data_selected_old.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)

    with open('../VAE/data_fold_%d_lod_%d.pkl' % (0, 5), 'rb') as pkl:
        data_train2 = pickle.load(pkl)

    batches = batch_provider(data_train2[:10000], 128, process_batch, report_progress=True)

    i = 0
    w_avr = None
    with torch.no_grad():
        for x in batches:
            encoder.eval()
            decoder.eval()

            styless = encoder.encode(x, layer_count - 1)
            w = encoder.style_encode(styless, layer_count - 1)
            if w_avr is None:
                w_avr = w
            else:
                if w.shape[0] == w_avr.shape[0]:
                    w_avr += w
            i +=1

    w_avr /= i
    w_avr = w_avr.mean(dim=0)

    im_size = 128
    im_count = len(ids)

    xs = process_batch([data_train[x] for x in ids])

    styless = encoder.encode(xs, layer_count - 1)
    w = encoder.style_encode(styless, layer_count - 1)

    mul = bimpy.Float()
    mul.value = 1.0

    mulm = bimpy.Float()
    mulm.value = 1.0

    addm = bimpy.Float()
    addm.value = 0.0

    mulv = bimpy.Float()
    mulv.value = 1.0

    addv = bimpy.Float()
    addv.value = 0.0

    layer_start = bimpy.Int()
    layer_end = bimpy.Int()
    layer_start.value = 0
    layer_end.value = 4

    def update_image():
        w_ = (w - w_avr) * mul.value + w_avr

        #mu, logvar = torch.split(w_, [vae.latent_size, vae.latent_size], dim=1)

        mu, logvar = torch.split(w, [decoder.latent_size, decoder.latent_size], dim=1)
        mu_, logvar_ = torch.split(w_, [decoder.latent_size, decoder.latent_size], dim=1)
        mu_avr, logvar_avr = torch.split(w_avr, [decoder.latent_size, decoder.latent_size], dim=0)

        styless_orig = decoder.style_decode(mu, layer_count - 1)
        styless = decoder.style_decode(mu_, layer_count - 1)
        styless_avr = decoder.style_decode(mu_avr, layer_count - 1)

        cut_layer_b = layer_start.value
        cut_layer_e = layer_end.value

        styless_new = []
        i = 0
        for style, style_avr, style_orig in zip(styless, styless_avr, styless_orig):
            if i > cut_layer_b and i < cut_layer_e:
                mu, var = torch.split(style, [style.shape[1] // 2, style.shape[1] // 2], dim=1)
                mu_avr, var_avr = torch.split(style_avr.view(1, -1), [style.shape[1] // 2, style.shape[1] // 2], dim=1)
                mu = (mu - mu_avr) * mulm.value + mu_avr
                mu = mu + addm.value
                var = (var - var_avr) * mulv.value + var_avr
                var = var + addv.value
                style = torch.cat((mu, var), dim=1)
                styless_new.append(style)
            else:
                styless_new.append(style_orig)
            i += 1

        styless = styless_new

        recs = decoder.decode(styless, layer_count - 1, True)

        im_width = math.ceil(pow(im_count, 0.5) + 0.5)
        im_height = math.ceil(im_count / im_width)
        canvas1 = np.zeros([3, im_size * (im_height), im_size * (im_width)])
        canvas2 = np.zeros([3, im_size * (im_height), im_size * (im_width)])

        for i in range(im_height):
            for j in range(im_width):
                if i * im_width + j < len(xs):
                    place(canvas1, xs[i * im_width + j], j, i)
                    place(canvas2, recs[i * im_width + j], j, i)

        canvas1 = np.clip(canvas1.transpose([1, 2, 0]) * 255, 0, 255).astype(np.uint8)
        canvas2 = np.clip(canvas2.transpose([1, 2, 0]) * 255, 0, 255).astype(np.uint8)
        return np.concatenate((canvas1, canvas2), axis=1)

    ctx = bimpy.Context()

    ctx.init(1800, 1600, "Styles")

    im = bimpy.Image(update_image())

    str = bimpy.String()

    while(not ctx.should_close()):
        with ctx:
            if bimpy.slider_int('Crop, start layer', layer_start, 0, 20):
                im = bimpy.Image(update_image())
            if bimpy.slider_int('Crop, end layer', layer_end, 0, 20):
                im = bimpy.Image(update_image())

            if bimpy.slider_float('mul', mul, -3.0, 3.0):
                im = bimpy.Image(update_image())

            if bimpy.slider_float('mulm', mulm, -3.0, 3.0):
                im = bimpy.Image(update_image())

            if bimpy.slider_float('mulv', mulv, -3.0, 3.0):
                im = bimpy.Image(update_image())

            if bimpy.slider_float('addm', addm, -3.0, 3.0):
                im = bimpy.Image(update_image())

            if bimpy.slider_float('addv', addv, -3.0, 3.0):
                im = bimpy.Image(update_image())

            bimpy.image(im)

            bimpy.set_window_font_scale(3.0)
            bimpy.input_text("", str, 255)
            bimpy.set_window_font_scale(2.0)


if __name__ == '__main__':
    main("encoder.pkl", "decoder.pkl")
