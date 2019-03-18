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

def load(model, filename):
    try:
        model.load_state_dict(torch.load(filename))
    except RuntimeError:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(filename))
        model = model.module
    model.eval()
    return model

def main():
    layer_count = 6
    latent_size = 128
    generator = Generator(maxf=128, layer_count=layer_count, latent_size=latent_size, channels=3)
    generator.cuda()
    generator = load(generator, "generator.pkl")

    mapping = Mapping(num_layers=2 * layer_count, latent_size=latent_size, dlatent_size=latent_size, mapping_fmaps=latent_size)
    mapping.cuda()
    mapping = load(mapping, "mapping.pkl")
    
    print("Trainable parameters:")
    count_parameters(generator)
    count_parameters(mapping)
    
    im_size = 128
    im_count = 16

    styles_avg = []
    
    sample = torch.randn(1024 * 4, latent_size).view(-1, latent_size)
    styles = list(mapping(sample))

    for s in styles:
        styles_avg.append(s.mean(dim=0, keepdim=True))
        
    sample = torch.randn(64, latent_size).view(-1, latent_size)
    styles = list(mapping(sample))
    
    styles_truct = []
    for s, a in zip(styles, styles_avg):
        m = 0.6
        styles_truct.append(s * m + a * (1.0 - m))
        
    styles = styles_truct
        
    print("Style count: %d" % len(styles))
            
    im_count = 8
    for t in range(16):
        canvas = np.zeros([3, im_size * (im_count), im_size * (im_count)])  
        rec = generator.decode(styles, layer_count - 1, 0.6)
        for i in range(im_count):
            for j in range(im_count):
                place(canvas, rec[i * im_count + j], i, j)  
        save_image(torch.Tensor(canvas), 'gen_%f.png' % t)  
    
    im_count = 16
    canvas = np.zeros([3, im_size * (im_count + 2), im_size * (im_count + 2)])    
    
    cut_layer_b = 0    
    cut_layer_e = 6

    for i in range(im_count):    
        #torch.cuda.manual_seed_all(1000 + i)
        style = [x[i] for x in styles] 
        rec = generator.decode(style, layer_count - 1, 0.7)  
        place(canvas, rec[0], 1, 2 + i)
        
        place(canvas, rec[0], 2 + i, 1)

    for i in range(im_count):
        for j in range(im_count):
            style_a = [x[i] for x in styles[:cut_layer_b]]    
            style_b = [x[j] for x in styles[cut_layer_b:cut_layer_e]]    
            style_c = [x[i] for x in styles[cut_layer_e:]]    
            style = style_a + style_b + style_c
            #torch.cuda.manual_seed_all(1000 + i)
            rec = generator.decode(style, layer_count - 1, 0.7)    
            place(canvas, rec[0], 2 + i, 2 + j)    
        
    save_image(torch.Tensor(canvas), 'reconstruction.png')    

if __name__ == '__main__':
    main()
