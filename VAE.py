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
import sys
sys.path.append('../PerceptualSimilarity')
from models import dist_model as dm

im_size = 128
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True,version='0.1')


def save_model(x, name):
    if isinstance(x, nn.DataParallel):
        torch.save(x.module.state_dict(), name)
    else:
        torch.save(x.state_dict(), name)


def loss_function(recon_x, x, lod):
    #return torch.mean((recon_x - x)**2)
    if lod > 1:
        if lod != 5:
            d = model.forward(F.interpolate(recon_x, scale_factor=2), F.interpolate(x, scale_factor=2), False)
        else:
            d = model.forward(recon_x, x, False)
        return d.mean() + torch.mean((recon_x - x)**2)
    else:
        return torch.mean((recon_x - x)**2)


def process_batch(batch):
    data = [x.transpose((2, 0, 1)) for x in batch]
    x = torch.tensor(np.asarray(data, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
    return x

if torch.cuda.device_count() == 4:
    #              4x4  8x8 16x16  32x32  64x64  128x128
    lod_2_batch = [512, 256, 128,   128,    128,     128]
if torch.cuda.device_count() == 3:
    #              4x4  8x8 16x16  32x32  64x64  128x128
    lod_2_batch = [512, 256, 128,   128,    128,     128]
elif torch.cuda.device_count() == 2:
    #              4x4  8x8 16x16  32x32  64x64  128x128
    lod_2_batch = [512, 256, 128,   128,    64,     32]
elif torch.cuda.device_count() == 1:
    #              4x4  8x8 16x16  32x32  64x64  128x128
    lod_2_batch = [512, 256, 128,   128,    64,     32]


def D_logistic_simplegp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real)).mean()

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1,2,3])
        loss = loss + r1_penalty.mean() * (r1_gamma * 0.5)
    return loss

    
def G_logistic_nonsaturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


def main(parallel=False):
    layer_count = 6
    epochs_per_lod = 15
    latent_size = 256

    autoencoder = Autoencoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    autoencoder.cuda()
    autoencoder.train()
    autoencoder.weight_init(mean=0, std=0.02)

    autoencoder.load_state_dict(torch.load("autoencoder.pkl"))

    print("Trainable parameters autoencoder:")
    count_parameters(autoencoder)

    if parallel:
        autoencoder = nn.DataParallel(autoencoder)
        autoencoder.layer_to_resolution = autoencoder.module.layer_to_resolution

    lr = 0.0002
    lr2 = 0.0005

    autoencoder_optimizer = optim.Adam([
        {'params': autoencoder.parameters()},
    ], lr=lr, betas=(0.9, 0.999), weight_decay=0)

    train_epoch = 100

    with open('data_selected_old.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)
        sample = process_batch(data_train[:32])
        del data_train

    lod = -1
    in_transition = False

    for epoch in range(75, train_epoch):
        autoencoder.train()

        new_lod = min(layer_count - 1, epoch // epochs_per_lod)
        if new_lod != lod:
            lod = new_lod
            print("#" * 80, "\n# Switching LOD to %d" % lod, "\n" + "#" * 80)
            print("Start transition")
            getattr(autoencoder.module, "decode_block%d" % (lod + 1)).noise_weight_1.data.normal_(0.1, 0.02)
            getattr(autoencoder.module, "decode_block%d" % (lod + 1)).noise_weight_2.data.normal_(0.1, 0.02)
            in_transition = True

        new_in_transition = (epoch % epochs_per_lod) < (epochs_per_lod // 2) and lod > 0 and epoch // epochs_per_lod == lod
        if new_in_transition != in_transition:
            in_transition = new_in_transition
            print("#" * 80, "\n# Transition ended", "\n" + "#" * 80)

        with open('../VAE/data_fold_%d_lod_%d.pkl' % (epoch % 5, lod), 'rb') as pkl:
            data_train = pickle.load(pkl)

        print("Train set size:", len(data_train))
        data_train = data_train[:4 * (len(data_train) // 4)]

        random.shuffle(data_train)

        batches = batch_provider(data_train, lod_2_batch[lod], process_batch, report_progress=True)

        rec_loss = []
        kl_loss = []

        epoch_start_time = time.time()

        # if (epoch + 1) == 40:
        #     autoencoder_optimizer.param_groups[0]['lr'] = lr / 2
        #     #discriminator_optimizer.param_groups[0]['lr'] = lr2 / 4
        #     print("learning rate change!")

        i = 0
        for x_orig in batches:
            if x_orig.shape[0] != lod_2_batch[lod]:
                continue
            autoencoder.train()
            autoencoder.zero_grad()

            blend_factor = float((epoch % epochs_per_lod) * len(data_train) + i) / float(epochs_per_lod // 2 * len(data_train))
            if not in_transition:
                blend_factor = 1

            needed_resolution = autoencoder.layer_to_resolution[lod]
            x = x_orig

            if in_transition:
                needed_resolution_prev = autoencoder.layer_to_resolution[lod - 1]
                x_prev = F.avg_pool2d(x_orig, 2, 2)
                x_prev_2x = F.interpolate(x_prev, needed_resolution)
                x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            autoencoder.zero_grad()
            rec = autoencoder(x, lod, blend_factor)
            loss_re = loss_function(rec, x, lod)
            rec_loss += [loss_re.item()]
            loss_re.backward()
            autoencoder_optimizer.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            
            def avg(lst): 
                if len(lst) == 0:
                    return 0
                return sum(lst) / len(lst) 
                
            # report losses and save samples each 60 iterations
            m = 7680 * 2
            i += lod_2_batch[lod]
            if i % m == 0:
                os.makedirs('results', exist_ok=True)
                rec_loss = avg(rec_loss)
                #kl_loss = avg(kl_loss)
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, rec_loss))

                rec_loss = []
                kl_loss = []
                with torch.no_grad():
                    autoencoder.eval()

                    sample_in = sample
                    while sample_in.shape[2] != needed_resolution:
                        sample_in = F.avg_pool2d(sample_in, 2, 2)

                    if in_transition:
                        needed_resolution_prev = autoencoder.layer_to_resolution[lod - 1]
                        sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
                        sample_in_prev_2x = F.interpolate(sample_in_prev, needed_resolution)
                        sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

                    rec = autoencoder(sample_in, lod, blend_factor)
                    rec = F.interpolate(rec, sample.shape[2])
                    sample_in = F.interpolate(sample_in, sample.shape[2])
                    resultsample = torch.cat([sample_in, rec], dim=0)
                    resultsample = (resultsample * 0.5 + 0.5).cpu()
                    save_image(resultsample,
                               'results/sample_' + str(epoch) + "_" + str(i // lod_2_batch[lod]) + '.jpg', nrow=8)
                    # w = list(mapping(sample))
                    # x_rec = autoencoder(w, lod, blend_factor)
                    # resultsample = x_rec * 0.5 + 0.5
                    # resultsample = resultsample.cpu()
                    # save_image(resultsample.view(-1, 3, needed_resolution, needed_resolution),
                    #            'results_rec/sample_' + str(epoch) + "_" + str(i // lod_2_batch[lod]) + '.png', nrow=8)
                    #x_rec = vae.decode(sample1)
                    #resultsample = x_rec * 0.5 + 0.5
                    #resultsample = resultsample.cpu()
                    #save_image(resultsample.view(-1, 3, im_size, im_size),
                    #           'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        del batches
        del data_train
        save_model(autoencoder, "autoencoder_tmp.pkl")
    print("Training finish!... save training results")
    save_model(autoencoder, "autoencoder.pkl")


if __name__ == '__main__':
    main(True)
