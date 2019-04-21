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
from torch.autograd import Function
import sys
sys.path.append('../PerceptualSimilarity')
from models import dist_model as dm
from tracker import LossTracker


im_size = 128
model = dm.DistModel()
model.initialize(model='net-lin',net='alex',use_gpu=True,version='0.1')

im_size = 128

def save_model(x, name):
    if isinstance(x, nn.DataParallel):
        torch.save(x.module.state_dict(), name)
    else:
        torch.save(x.state_dict(), name)


def loss_kl(mu, logvar):
    return -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))


def loss_rec(recon_x, x, lod):
    if lod > 2:
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
    lod_2_batch = [512, 256, 128,   128,    128,     64]
if torch.cuda.device_count() == 3:
    #              4x4  8x8 16x16  32x32  64x64  128x128
    lod_2_batch = [512, 256, 128,   128,    128,     64]
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


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


def main(parallel=False):
    layer_count = 6
    epochs_per_lod = 15
    latent_size = 256

    encoder = Encoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    decoder = Decoder(layer_count=layer_count, startf=64, maxf=256, latent_size=latent_size, channels=3)
    encoder.cuda()
    encoder.train()
    encoder.weight_init(mean=0, std=0.02)
    decoder.cuda()
    decoder.train()
    decoder.weight_init(mean=0, std=0.02)

    #autoencoder.load_state_dict(torch.load("autoencoder.pkl"))

    print("Trainable parameters encoder:")
    count_parameters(encoder)

    print("Trainable parameters decoder:")
    count_parameters(decoder)

    if parallel:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        decoder.layer_to_resolution = decoder.module.layer_to_resolution

    lr = 0.0002
    alpha = 0.15
    M = 0.25

    autoencoder_optimizer = optim.Adam([
        {'params': list(decoder.parameters()) + list(encoder.parameters())},
    ], lr=lr, betas=(0.9, 0.999), weight_decay=0)
    #
    # encoder_optimizer = optim.Adam([
    #     {'params': encoder.parameters()},
    # ], lr=lr, betas=(0.9, 0.999), weight_decay=0)
    #
    # decoder_optimizer = optim.Adam([
    #     {'params': decoder.parameters()},
    # ], lr=lr, betas=(0.9, 0.999), weight_decay=0)

    train_epoch = 100

    with open('data_selected_old.pkl', 'rb') as pkl:
        data_train = pickle.load(pkl)
        sample = process_batch(data_train[:32])
        del data_train

    lod = -1
    in_transition = False

    samplew = torch.randn(64, latent_size).view(-1, latent_size)

    tracker = LossTracker()

    Lae_loss = tracker.add('Lae')
    Ladv_loss = tracker.add('Ladv')
    LklZ_loss = tracker.add('LklZ')

    for epoch in range(train_epoch):
        encoder.train()
        decoder.train()

        new_lod = min(layer_count - 1, epoch // epochs_per_lod)
        if new_lod != lod:
            lod = new_lod
            print("#" * 80, "\n# Switching LOD to %d" % lod, "\n" + "#" * 80)
            print("Start transition")
            getattr(decoder.module, "decode_block%d" % (lod + 1)).noise_weight_1.data.normal_(0.1, 0.02)
            getattr(decoder.module, "decode_block%d" % (lod + 1)).noise_weight_2.data.normal_(0.1, 0.02)
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

        epoch_start_time = time.time()

        if (epoch + 1) == 50:
            autoencoder_optimizer.param_groups[0]['lr'] = lr / 4
            #discriminator_optimizer.param_groups[0]['lr'] = lr2 / 4
            print("learning rate change!")
        if (epoch + 1) == 90:
            autoencoder_optimizer.param_groups[0]['lr'] = lr / 4 / 4
            #discriminator_optimizer.param_groups[0]['lr'] = lr2 / 4
            print("learning rate change!")

        i = 0
        for x_orig in batches:
            if x_orig.shape[0] != lod_2_batch[lod]:
                continue
            encoder.train()
            decoder.train()

            blend_factor = float((epoch % epochs_per_lod) * len(data_train) + i) / float(epochs_per_lod // 2 * len(data_train))
            if not in_transition:
                blend_factor = 1

            needed_resolution = decoder.layer_to_resolution[lod]
            x = x_orig

            if in_transition:
                needed_resolution_prev = decoder.layer_to_resolution[lod - 1]
                x_prev = F.avg_pool2d(x_orig, 2, 2)
                x_prev_2x = F.interpolate(x_prev, needed_resolution)
                x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            Z = encoder(x, lod, blend_factor)

            Xr = decoder(*Z, lod, blend_factor)

            Lae = loss_rec(Xr, x, lod)

            LklZ = loss_kl(*Z)

            loss1 = LklZ * 0.03 + Lae

            Zr = encoder(grad_reverse(Xr), lod, blend_factor)

            Ladv = -loss_kl(*Zr) * alpha

            loss2 = Ladv * 0.03

            autoencoder_optimizer.zero_grad()
            (loss1 + loss2).backward()
            autoencoder_optimizer.step()

            Lae_loss << Lae
            Ladv_loss << Ladv
            LklZ_loss << LklZ

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
                os.makedirs('results_gen', exist_ok=True)

                print('\n[%d/%d] - ptime: %.2f, %s' % (
                    (epoch + 1), train_epoch, per_epoch_ptime, str(tracker)))

                tracker.register_means(epoch + i / len(batches) / lod_2_batch[lod])
                tracker.plot()

                with torch.no_grad():
                    encoder.eval()
                    decoder.eval()

                    sample_in = sample
                    while sample_in.shape[2] != needed_resolution:
                        sample_in = F.avg_pool2d(sample_in, 2, 2)

                    if in_transition:
                        needed_resolution_prev = decoder.layer_to_resolution[lod - 1]
                        sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
                        sample_in_prev_2x = F.interpolate(sample_in_prev, needed_resolution)
                        sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

                    Z = encoder(sample_in, lod, blend_factor)
                    rec = decoder(*Z, lod, blend_factor)
                    rec = F.interpolate(rec, sample.shape[2])
                    sample_in = F.interpolate(sample_in, sample.shape[2])
                    resultsample = torch.cat([sample_in, rec], dim=0)
                    resultsample = (resultsample * 0.5 + 0.5).cpu()
                    save_image(resultsample,
                               'results/sample_' + str(epoch) + "_" + str(i // lod_2_batch[lod]) + '.jpg', nrow=8)

                    x_rec = decoder(samplew, None, lod, blend_factor)

                    x_rec = F.interpolate(x_rec, sample.shape[2])
                    resultsample = (x_rec * 0.5 + 0.5).cpu()
                    save_image(resultsample,
                               'results_gen/sample_' + str(epoch) + "_" + str(i // lod_2_batch[lod]) + '.jpg', nrow=8)

        del batches
        del data_train
        save_model(encoder, "encoder_tmp.pkl")
        save_model(decoder, "decoder_tmp.pkl")
    print("Training finish!... save training results")
    save_model(encoder, "encoder.pkl")
    save_model(decoder, "decoder.pkl")


if __name__ == '__main__':
    main(True)
