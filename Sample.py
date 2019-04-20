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

im_size = 128


def process_batch(batch):
    data = [misc.imresize(x, [im_size, im_size]).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).cuda() / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x


def place(canvas, image, x, y):
    image = image.cpu().detach().numpy()
    im_size = image.shape[1]
    canvas[:, y * im_size : (y + 1) * im_size, x * im_size : (x + 1) * im_size] = image * 0.5 + 0.5


def main(model_filename):
    layer_count = 6
    latent_size = 256
    vae = Autoencoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    vae.cuda()
    try:
        vae.load_state_dict(torch.load(model_filename))
    except RuntimeError:
        vae = nn.DataParallel(vae)
        vae.load_state_dict(torch.load(model_filename))
        vae = vae.module
    vae.eval()

    print("Trainable parameters:")
    count_parameters(vae)
    
    with open('data_selected.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)

        im_size = 128

        
        #    Bold man with glasses
        #    Blond yang woman
        #    Kid
        #    Woman with pose different than frontal and smiling
        #    Older man smiling

        #    Dark hair smiling yang wonam
        #    Blond smiling yang woman with non frontal pose
        #    Black young woman
        #    Brown color (Asian?) girl
        #    Yang man hairy
        #   Smiling brown hair woman
    
        ids = [18, 2, 10, 1, 13, 4]
        idsd = [3, 4, 14, 21, 20, 19]
        
        #ids = [i for i in range(len(data_train))]
        im_count = max(len(ids), len(idsd))
        
        xs = process_batch([data_train[x] for x in ids])
        xd = process_batch([data_train[x] for x in idsd])
        #x = process_batch(data_train[im_count * 2:im_count * 3])

        styless = vae.encode(xs, layer_count - 1)
        w = vae.style_encode(styless, layer_count - 1)
        styless = vae.style_decode(w, layer_count - 1)

        recs = vae.decode(styless, layer_count - 1, True)
        stylesd = vae.encode(xd, layer_count - 1)
        w = vae.style_encode(stylesd, layer_count - 1)
        stylesd = vae.style_decode(w, layer_count - 1)

        recd = vae.decode(stylesd, layer_count - 1, True)

        canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])

        for i in range(im_count):
            place(canvas, xs[i], 0, 2 + i)
            save_image(xs[i] * 0.5 + 0.5, 'source_%d.jpg' % i)
            
            place(canvas, recs[i], 1, 2 + i)

            place(canvas, xd[i], 2 + i, 0)
            place(canvas, recd[i], 2 + i, 1)

        cut_layer_b = len(styless)-1 - 9
        cut_layer_e = len(styless)-1 - 14
        
        for i in [0, 5, 2]:
            for j in range(im_count):
                style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
                style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
                style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
                style = style_a + style_b + style_c
                
                save_image(xd[i] * 0.5 + 0.5, 'dst_coarse_%d.jpg' % i)
                
                rec = vae.decode(style, layer_count - 1, True)
                save_image(rec * 0.5 + 0.5, 'rec_coarse_%d_%d.jpg'% (i, j))
                #place(canvas, rec[0], 2 + i, 2 + j)

        cut_layer_b = len(styless)-1 - 4
        cut_layer_e = len(styless)-1 - 10
        
        for i in [3, 4]:
            for j in range(im_count):
                style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
                style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
                style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
                style = style_a + style_b + style_c
                
                save_image(xd[i] * 0.5 + 0.5, 'dst_mid_%d.jpg' % i)
                
                rec = vae.decode(style, layer_count - 1, True)
                save_image(rec * 0.5 + 0.5, 'rec_mid_%d_%d.jpg' % (i, j))
                #place(canvas, rec[0], 2 + i, 2 + j)

        cut_layer_b = len(styless)-1 - 0
        cut_layer_e = len(styless)-1 - 4
        
        for i in [1]:
            for j in range(im_count):
                style_a = [x[i].unsqueeze(0) for x in stylesd[:cut_layer_b]]
                style_b = [x[j].unsqueeze(0) for x in styless[cut_layer_b:cut_layer_e]]
                style_c = [x[i].unsqueeze(0) for x in stylesd[cut_layer_e:]]
                style = style_a + style_b + style_c
                
                save_image(xd[i] * 0.5 + 0.5, 'dst_fine_%d.jpg' % i)
                
                rec = vae.decode(style, layer_count - 1, True)
                save_image(rec * 0.5 + 0.5, 'rec_fine_%d.jpg' % j)
                #place(canvas, rec[0], 2 + i, 2 + j)

        save_image(torch.Tensor(canvas), 'reconstruction.png')

        canvas = np.zeros([3, im_size * (im_count), im_size * (im_count)])
        
        for i in range(im_count):
            i_ = im_count - i - 1
            for j in range(im_count):
                style_a = [(x[0][i].unsqueeze(0), x[1][i].unsqueeze(0)) for x in styless]
                style_b = [(x[0][i_].unsqueeze(0), x[1][i_].unsqueeze(0)) for x in stylesd]
                
                v = j / (im_count-1)
                v_ = 1. - v
                style = [(a[0] * v + b[0] * v_, a[1] * v + b[1] * v_) for a, b in zip(style_a, style_b)]
                
                rec = vae.decode(style, layer_count - 1, True)
                place(canvas, rec[0], j, i)

        save_image(torch.Tensor(canvas), 'blending.png')
        
        canvas = np.zeros([3, im_size * (im_count), im_size * (im_count)])
        
        style_a = [(x[0][0].unsqueeze(0), x[1][0].unsqueeze(0)) for x in styless]
        style_b = [(x[0][1].unsqueeze(0), x[1][1].unsqueeze(0)) for x in stylesd]
        style_c = [(x[0][2].unsqueeze(0), x[1][2].unsqueeze(0)) for x in styless]
        style_d = [(x[0][3].unsqueeze(0), x[1][3].unsqueeze(0)) for x in stylesd]
        
        for i in range(im_count):
            i_ = im_count - i - 1
            for j in range(im_count):
                j_ = im_count - i - 1
                
                iv = i / (im_count-1)
                jv = j / (im_count-1)
                
                iv_ = 1. - iv
                jv_ = 1. - jv
                styleab = [(a[0] * iv + b[0] * iv_, a[1] * iv + b[1] * iv_) for a, b in zip(style_a, style_b)]
                stylecd = [(c[0] * iv + d[0] * iv_, c[1] * iv + d[1] * iv_) for c, d in zip(style_c, style_d)]
                
                style = [(a[0] * jv + b[0] * jv_, a[1] * jv + b[1] * jv_) for a, b in zip(styleab, stylecd)]
                
                rec = vae.decode(style, layer_count - 1, True)
                place(canvas, rec[0], j, i)

        save_image(torch.Tensor(canvas), 'bilinear.png')
        
        
        del data_train


if __name__ == '__main__':
    main("autoencoder.pkl")
