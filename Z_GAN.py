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
from torchvision.utils import save_image
from net import *
import numpy as np
import pickle
import time
import random
import os
import utils
from tracker import LossTracker
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from torch.optim import Adam
from dataloader import *
from tqdm import tqdm
from dlutils import batch_provider
from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
import math

# from model_ae_minist import Model
from model_z_gan import Model

from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
import gc
from sklearn import metrics
from collections import OrderedDict
from sklearn import svm


def save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cfg, encoder_optimizer, decoder_optimizer):
    os.makedirs('results', exist_ok=True)

    logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        lod2batch.get_blend_factor(),
        encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()

        needed_resolution = model.decoder.layer_to_resolution[lod2batch.lod]
        sample_in = sample
        #while sample_in.shape[2] != needed_resolution:
        #    sample_in = F.avg_pool2d(sample_in, 2, 2)

        blend_factor = lod2batch.get_blend_factor()
        if lod2batch.in_transition:
            needed_resolution_prev = model.decoder.layer_to_resolution[lod2batch.lod - 1]
            sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
            sample_in_prev_2x = F.interpolate(sample_in_prev, needed_resolution)
            sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

        Z, _ = model.encode(sample_in, lod2batch.lod, blend_factor)

        # Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        # rec1 = model.decoder(Z, lod2batch.lod, blend_factor, noise=False)
        rec2 = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

        # rec1 = F.interpolate(rec1, sample.shape[2])
        rec2 = F.interpolate(rec2, sample.shape[2])
        sample_in = F.interpolate(sample_in, sample.shape[2])

        Z = model.mapping_fl(samplez)[:, 0]
        # Z = F.normalize(Z)
        g_rec = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)
        g_rec = F.interpolate(g_rec, sample.shape[2])

        resultsample = torch.cat([sample_in, rec2, g_rec], dim=0)

        @utils.async_func
        def save_pic(x_rec):
            result_sample = x_rec
            result_sample = result_sample.cpu()
            f = os.path.join(cfg.OUTPUT_DIR,
                                                   'sample_%d_%d.jpg' % (
                                                       lod2batch.current_epoch + 1,
                                                       lod2batch.iteration // 1000)
                                                   )
            print("Saved to %s" % f)
            save_image(result_sample, f, nrow=16)
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())

            tracker.plot()

        save_pic(resultsample)


def gpu_nnc_predict(trX, trY, teX, batch_size=256):
    metric_fn = F.pairwise_distance
    idxs = []
    for i in range(0, len(teX), batch_size):
        mb_dists = []
        mb_idxs = []
        for j in range(0, len(trX), batch_size):
            v1 = teX[i:i+batch_size][:, None, ...]
            v2 = trX[j:j+batch_size][None, :, ...]
            #dist = (torch.sum(torch.abs(v1 - v2), dim=2) ** 0.5).detach().cpu().numpy()
            dist = (torch.sum((v1 - v2)**2, dim=2) ** 0.5).detach().cpu().numpy()

            mb_dists.append(np.min(dist, axis=1))
            mb_idxs.append(j + np.argmin(dist, axis=1))
        mb_idxs = np.stack(mb_idxs)
        mb_dists = np.stack(mb_dists)
        i = mb_idxs[np.argmin(mb_dists, axis=0), np.arange(mb_idxs.shape[1])]
        idxs.append(i)
    idxs = np.concatenate(idxs, axis=0)
    nearest = np.asarray(trY)[idxs]
    return nearest


def eval(cfg, logger, encoder, do_svm=False):
    local_rank = 0
    world_size = 1
    dataset_train = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=True, needs_labels=True)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS, train=False, needs_labels=True)

    encoder.eval()

    batch_size = cfg.TRAIN.LOD_2_BATCH_1GPU[len(cfg.TRAIN.LOD_2_BATCH_1GPU) - 1]

    dataset_train.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)
    dataset_test.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, batch_size)

    batches_train = make_dataloader_y(cfg, logger, dataset_train, batch_size, 0)
    batches_test = make_dataloader_y(cfg, logger, dataset_test, batch_size, 0)

    gc.collect()

    # @utils.cache
    def compute_train():
        train_X = []
        train_X2 = []
        train_Y = []

        for x_orig, y in tqdm(batches_train):
            with torch.no_grad():
                x = (x_orig / 255)

                Z, E = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, report_feature=True)
                train_X += torch.split(Z, 1)
                train_X2 += torch.split(E, 1)
                train_Y += list(y)

        train_X2 = torch.cat(train_X2)
        train_X = torch.cat(train_X)
        return train_X, train_X2, train_Y

    # @utils.cache
    def compute_test():
        test_X = []
        test_X2 = []
        test_Y = []

        for x_orig, y in tqdm(batches_test):
            with torch.no_grad():
                x = (x_orig / 255)

                Z, E = encoder(x, cfg.DATASET.MAX_RESOLUTION_LEVEL, 1, report_feature=True)
                test_X += torch.split(Z, 1)
                test_X2 += torch.split(E, 1)
                test_Y += list(y)
        test_X = torch.cat(test_X)
        test_X2 = torch.cat(test_X2)
        return test_X, test_X2, test_Y

    train_X, train_X2, train_Y = compute_train()
    test_X, test_X2, test_Y = compute_test()

    train_Y = np.asarray(train_Y)
    test_Y = np.asarray(test_Y)

    prediction = gpu_nnc_predict(train_X, train_Y, test_X)
    prediction_f = gpu_nnc_predict(train_X2, train_Y, test_X2)

    acc = metrics.accuracy_score(test_Y, prediction)
    acc_f = metrics.accuracy_score(test_Y, prediction_f)

    # logger.info("*" * 100)
    # logger.info("ACCURACY Embedding space: %f" % acc)
    # logger.info("ACCURACY Feature space: %f" % acc_f)
    # logger.info("*" * 100)

    outs = OrderedDict()

    outs['NNC_e'] = acc * 100.
    outs['NNC_e-'] = acc_f * 100.

    if do_svm:
        s = svm.LinearSVC(max_iter=5000, C=0.02)

        s.fit(train_X.cpu(), train_Y)
        prediction = s.predict(test_X.cpu())

        acc = metrics.accuracy_score(test_Y, prediction)

        s.fit(train_X2.cpu(), train_Y)
        prediction = s.predict(test_X2.cpu())

        acc_f = metrics.accuracy_score(test_Y, prediction)

        outs['SVM_e'] = acc * 100.
        outs['SVM_e-'] = acc_f * 100.

    def format_str(key):
        def is_prop(key, prop_metrics=['NNC','SVM', 'CLS']):
            return any(key.startswith(m) for m in prop_metrics)
        return '%s: %.2f' + ('%%' if is_prop(key) else '')
    logger.info('  '.join(format_str(k) % (k, v)
                    for k, v in outs.items()))


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
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
        mapping_to_latent=cfg.MODEL.MAPPING_TO_LATENT,
        mapping_from_latent=cfg.MODEL.MAPPING_FROM_LATENT
    )
    model.cuda(local_rank)
    model.train()

    if local_rank == 0:
        model_s = Model(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER,
            mapping_to_latent=cfg.MODEL.MAPPING_TO_LATENT,
            mapping_from_latent=cfg.MODEL.MAPPING_FROM_LATENT
        )
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None

        decoder = model.module.decoder
        encoder = model.module.encoder
        mapping_tl = model.module.mapping_tl
        mapping_fl = model.module.mapping_fl
        dlatent_avg = model.module.dlatent_avg
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_tl = model.mapping_tl
        mapping_fl = model.mapping_fl
        dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    LREQ = False

    if LREQ:
        decoder_optimizer = LREQAdam([
            {'params': decoder.parameters()},
            {'params': mapping_fl.parameters()}
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0.)

        encoder_optimizer = LREQAdam([
            {'params': encoder.parameters()},
            {'params': mapping_tl.parameters()},
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0.)

    else:
        decoder_optimizer = Adam([
            {'params': decoder.parameters()},
            {'params': mapping_fl.parameters(), 'lr': 0.00002}
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0.)

        encoder_optimizer = Adam([
            {'params': encoder.parameters()},
            {'params': mapping_tl.parameters(), 'lr': 0.00002},
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0.)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'mapping_fl': mapping_fl,
        'dlatent_avg': dlatent_avg
    }
    if local_rank == 0:
        model_dict['discriminator_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        model_dict['mapping_tl_s'] = model_s.mapping_tl
        model_dict['mapping_fl_s'] = model_s.mapping_fl

    tracker = LossTracker(cfg.OUTPUT_DIR)

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()#file_name='results_ae/model_tmp.pth')
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size)

    mnist = True
    if mnist:
        dlutils.download.mnist()
        mnist = dlutils.reader.Mnist('mnist').items
        mnist = np.asarray([x[1] for x in mnist], np.float32)

        def process_batch(batch):
            x = torch.tensor(np.asarray(batch, dtype=np.float32), requires_grad=True).cuda()
            # x = F.pad(torch.tensor(x).view(x.shape[0], 1, 28, 28), (2, 2, 2, 2)) / 127.5 - 1.
            x = torch.tensor(x).view(x.shape[0], 1, 28, 28) / 255
            return x.detach()

        sample = process_batch(mnist[:32])
    else:
        # with open('data_selected_old.pkl', 'rb') as pkl:
        #     data_train = pickle.load(pkl)
        #
        #     def process_batch(batch):
        #         data = [x.transpose((2, 0, 1)) for x in batch]
        #         x = torch.tensor(np.asarray(data, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
        #         return x
        #     sample = process_batch(data_train[:32])
        #     del data_train

        path = 'realign1024x1024'
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(path + '/' + filename))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 255
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
            sample = torch.stack(src)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
                                                                lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                lod2batch.lod,
                                                                2 ** lod2batch.get_lod_power2(),
                                                                2 ** lod2batch.get_lod_power2(),
                                                                lod2batch.get_blend_factor(),
                                                                len(dataset) * world_size))

        dataset.reset(lod2batch.get_lod_power2(), lod2batch.get_per_GPU_batch_size())
        batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        alt = False

        i = 0
        gc.collect()

        with torch.autograd.profiler.profile(use_cuda=True, enabled=False) as prof:
            for x_orig in tqdm(batches):
                i +=1
                with torch.no_grad():
                    if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                        continue
                    if need_permute:
                        x_orig = x_orig.permute(0, 3, 1, 2)
                    x_orig = (x_orig / 255)

                    blend_factor = lod2batch.get_blend_factor()

                    needed_resolution = layer_to_resolution[lod2batch.lod]
                    x = x_orig

                    if lod2batch.in_transition:
                        needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                        x_prev = F.avg_pool2d(x_orig, 2, 2)
                        x_prev_2x = F.interpolate(x_prev, needed_resolution)
                        x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

                x.requires_grad = True

                encoder_optimizer.zero_grad()
                loss_d = model(x, lod2batch.lod, blend_factor, d_train=True, ae=False, alt=False)
                tracker.update(dict(loss_d=loss_d))
                loss_d.backward()
                encoder_optimizer.step()

                decoder_optimizer.zero_grad()
                loss_g = model(x, lod2batch.lod, blend_factor, d_train=False, ae=False, alt=False)
                tracker.update(dict(loss_g=loss_g))
                loss_g.backward()
                decoder_optimizer.step()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True, alt=False)
                tracker.update(dict(lae=lae))
                (lae).backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                #
                # encoder_optimizer.zero_grad()
                # decoder_optimizer.zero_grad()
                # lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True, alt=True)
                # tracker.update(dict(lae=lae))
                # (lae).backward()
                # encoder_optimizer.step()
                # decoder_optimizer.step()

                if local_rank == 0:
                    betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                    model_s.lerp(model, betta)

                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time

                lod_for_saving_model = lod2batch.lod
                lod2batch.step()
                if local_rank == 0:
                    if lod2batch.is_time_to_save():
                        checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                    if lod2batch.is_time_to_report():
                        save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cfg, encoder_optimizer, decoder_optimizer)

        scheduler.step()
        save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cfg, encoder_optimizer, decoder_optimizer)

        if epoch % 100 == 0 or epoch == cfg.TRAIN.TRAIN_EPOCHS - 1:
            with torch.no_grad():
                if local_rank == 0:
                    checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)

                encoder.eval()
                eval(cfg, logger, encoder=encoder)
                encoder.train()

    with torch.no_grad():
        encoder.eval()
        eval(cfg, logger, encoder=encoder, do_svm=True)
        encoder.train()

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_z.yaml',
        world_size=gpu_count)
