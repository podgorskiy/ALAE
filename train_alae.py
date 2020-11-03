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
import json
import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader_ecog import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
import numpy as np
from torch import autograd
from ECoGDataSet import concate_batch
def save_sample(lod2batch, tracker, sample, samplez, samplez_global, x, logger, model, cfg, encoder_optimizer, decoder_optimizer,filename=None,ecog=None,mask_prior=None,mode='test'):
    os.makedirs('results', exist_ok=True)
    logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        lod2batch.get_blend_factor(),
        encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    # sample = sample.transpose(-2,-1)
    with torch.no_grad():
        model.eval()
        # sample = sample[:lod2batch.get_per_GPU_batch_size()]
        # samplez = samplez[:lod2batch.get_per_GPU_batch_size()]

        needed_resolution = model.decoder.layer_to_resolution[lod2batch.lod]
        sample_in_all = torch.tensor([])
        rec1_all = torch.tensor([])
        rec2_all = torch.tensor([])
        g_rec_all = torch.tensor([])
        for i in range(0,sample.shape[0],9):
            sample_in = sample[i:np.minimum(i+9,sample.shape[0])]
            if ecog is not None:
                ecog_in = [ecog[j][i:np.minimum(i+9,sample.shape[0])] for j in range(len(ecog))]
                mask_prior_in = [mask_prior[j][i:np.minimum(i+9,sample.shape[0])] for j in range(len(mask_prior))]
            x_in = x[i:np.minimum(i+9,sample.shape[0])]
            samplez_in = samplez[i:np.minimum(i+9,sample.shape[0])]
            samplez_global_in = samplez_global[i:np.minimum(i+9,sample.shape[0])]
            while sample_in.shape[2] > needed_resolution:
                sample_in = F.avg_pool2d(sample_in, 2, 2)
            assert sample_in.shape[2] == needed_resolution

            blend_factor = lod2batch.get_blend_factor()
            if lod2batch.in_transition:
                needed_resolution_prev = model.decoder.layer_to_resolution[lod2batch.lod - 1]
                sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
                sample_in_prev_2x = F.interpolate(sample_in_prev, scale_factor=2)
                sample_in = sample_in * blend_factor + sample_in_prev_2x * (1.0 - blend_factor)

            Z, _ = model.encode(sample_in, lod2batch.lod, blend_factor)
            if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
                Z,Z_global = Z
            if cfg.MODEL.Z_REGRESSION:
                Z = model.mapping_fl(Z[:, 0])
            else:
                if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
                    Z = Z.repeat(1, model.mapping_fl.num_layers, 1,1)
                    Z_global = Z_global.repeat(1, model.mapping_fl.num_layers, 1)
                    Z = (Z, Z_global)
                else:
                    if cfg.MODEL.TEMPORAL_W:
                        Z = Z.repeat(1, model.mapping_fl.num_layers, 1,1)
                    else:
                        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)

            rec1 = model.decoder(Z, lod2batch.lod, blend_factor, noise=False)
            rec2 = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

            # rec1 = F.interpolate(rec1, sample.shape[2])
            # rec2 = F.interpolate(rec2, sample.shape[2])
            # sample_in = F.interpolate(sample_in, sample.shape[2])
                    
            if ecog is not None:
                Z = model.ecog_encoder(ecog = ecog_in, mask_prior = mask_prior_in)
                if cfg.MODEL.TEMPORAL_W and cfg.MODEL.GLOBAL_W:
                    Z, Z_global = Z
                    Z = Z.view(Z.shape[0], 1, Z.shape[1],Z.shape[2]).repeat(1, model.mapping_fl.num_layers, 1, 1)
                    Z_global = Z_global.view(Z_global.shape[0], 1, Z_global.shape[1]).repeat(1, model.mapping_fl.num_layers, 1)
                    Z = (Z,Z_global)
                else:
                    if cfg.MODEL.TEMPORAL_W:
                        Z = Z.view(Z.shape[0], 1, Z.shape[1],Z.shape[2]).repeat(1, model.mapping_fl.num_layers, 1, 1)
                    else:
                        Z = Z.view(Z.shape[0], 1, Z.shape[1]).repeat(1, model.mapping_fl.num_layers, 1)
            else:
                Z = model.mapping_fl(samplez_in,samplez_global_in)
            g_rec = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

            # g_rec = model.generate(lod2batch.lod, blend_factor, count=ecog_in[0].shape[0], z=samplez_in.detach(), z_global=samplez_global_in, noise=True,return_styles=False,ecog=ecog_in,mask_prior=mask_prior_in)


            # g_rec = F.interpolate(g_rec, sample.shape[2])
            sample_in_all = torch.cat([sample_in_all,sample_in],dim=0)
            rec1_all = torch.cat([rec1_all,rec1],dim=0)
            rec2_all = torch.cat([rec2_all,rec2],dim=0)
            g_rec_all = torch.cat([g_rec_all,g_rec],dim=0)

        print(mode+' suploss is',torch.mean((g_rec_all-sample_in_all).abs()))
        resultsample = torch.cat([sample_in_all, rec1_all, rec2_all, g_rec_all], dim=0)
        if cfg.DATASET.BCTS:
            resultsample = resultsample.transpose(-2,-1)

        # @utils.async_func
        def save_pic(x_rec):
            if mode=='test':
                tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            # tracker.plot()

            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            if filename:
                f =filename
            else:
                if mode == 'test':
                    f = os.path.join(cfg.OUTPUT_DIR,
                                    'sample_%d_%d.jpg' % (
                                        lod2batch.current_epoch + 1,
                                        lod2batch.iteration // 1000)
                                )
                else:
                    f = os.path.join(cfg.OUTPUT_DIR,
                                    'sample_train_%d_%d.jpg' % (
                                        lod2batch.current_epoch + 1,
                                        lod2batch.iteration // 1000)
                                )
            print("Saved to %s" % f)
            # save_image(result_sample, f, nrow=min(32, lod2batch.get_per_GPU_batch_size()))
            save_image(result_sample, f, nrow=x_rec.shape[0]//4)

        save_pic(resultsample)


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
        with_ecog = cfg.MODEL.ECOG,
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
            ecog_encoder=cfg.MODEL.MAPPING_FROM_ECOG,
            z_regression=cfg.MODEL.Z_REGRESSION,
            average_w = cfg.MODEL.AVERAGE_W,
            spec_chans = cfg.DATASET.SPEC_CHANS,
            temporal_samples = cfg.DATASET.TEMPORAL_SAMPLES,
            temporal_w = cfg.MODEL.TEMPORAL_W,
            global_w = cfg.MODEL.GLOBAL_W,
            temporal_global_cat = cfg.MODEL.TEMPORAL_GLOBAL_CAT,
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
            with_ecog = cfg.MODEL.ECOG,
        )
        model_s.cuda(local_rank)
        model_s.eval()
        model_s.requires_grad_(False)
    # print(model)
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
        ppl_mean = model.module.ppl_mean
        ppl_d_mean = model.module.ppl_d_mean
        if hasattr(model,'ecog_encoder'):
            ecog_encoder = model.module.ecog_encoder
        if cfg.MODEL.W_CLASSIFIER:
            mapping_tw = model.module.mapping_tw
    else:
        decoder = model.decoder
        encoder = model.encoder
        mapping_tl = model.mapping_tl
        mapping_fl = model.mapping_fl
        dlatent_avg = model.dlatent_avg
        ppl_mean = model.ppl_mean
        if hasattr(model,'ecog_encoder'):
            ecog_encoder = model.ecog_encoder
        ppl_d_mean = model.ppl_d_mean
        if cfg.MODEL.W_CLASSIFIER:
            mapping_tw = model.mapping_tw

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    if cfg.MODEL.ECOG:
        decoder_optimizer = LREQAdam([
            {'params': decoder.parameters()},
            {'params': ecog_encoder.parameters()}
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
    else:
        decoder_optimizer = LREQAdam([
            {'params': decoder.parameters()},
            {'params': mapping_fl.parameters()}
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    if cfg.MODEL.ECOG:
        ecog_encoder_optimizer = LREQAdam([
            {'params': ecog_encoder.parameters()}
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    if cfg.MODEL.W_CLASSIFIER:
        encoder_optimizer = LREQAdam([
            {'params': encoder.parameters()},
            {'params': mapping_tl.parameters()},
            {'params': mapping_tw.parameters()},
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
    else:
        encoder_optimizer = LREQAdam([
            {'params': encoder.parameters()},
            {'params': mapping_tl.parameters()},
        ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

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
        'dlatent_avg': dlatent_avg,
        'ppl_mean':ppl_mean,
        'ppl_d_mean':ppl_d_mean,
    }
    if hasattr(model,'ecog_encoder'):
        model_dict['ecog_encoder'] = ecog_encoder
    if local_rank == 0:
        model_dict['discriminator_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        model_dict['mapping_tl_s'] = model_s.mapping_tl
        model_dict['mapping_fl_s'] = model_s.mapping_fl
        if hasattr(model_s,'ecog_encoder'):
            model_dict['ecog_encoder_s'] = model_s.ecog_encoder

    tracker = LossTracker(cfg.OUTPUT_DIR)

    auxiliary = {'encoder_optimizer': encoder_optimizer,
                'decoder_optimizer': decoder_optimizer,
                'scheduler': scheduler,
                'tracker': tracker
                }
    if cfg.MODEL.ECOG:
        auxiliary['ecog_encoder_optimizer']=ecog_encoder_optimizer
    checkpointer = Checkpointer(cfg,
                                model_dict,
                                auxiliary,
                                logger=logger,
                                save=local_rank == 0)

    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=False,ignore_auxiliary=True,file_name='./training_artifacts/ecog_residual_cycle/model_tmp_lod4.pth')
    extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=True,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/ecog_residual_latent128_temporal_lesstemporalfeature_noprogressive_HBw_ppl_ppld_localreg_debug/model_tmp_lod6.pth')
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution
    with open('train_param.json','r') as rfile:
        param = json.load(rfile)
    # data_param, train_param, test_param = param['Data'], param['Train'], param['Test']
    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,param=param)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,train=False,param=param)

    rnd = np.random.RandomState(3456)
    # latents = rnd.randn(len(dataset_test.dataset), cfg.MODEL.LATENT_SPACE_SIZE)
    # samplez = torch.tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(cfg, logger, world_size, dataset_size=len(dataset) * world_size, progressive = (not(cfg.FINETUNE.FINETUNE) and cfg.TRAIN.PROGRESSIVE))

    if cfg.DATASET.SAMPLES_PATH:
        path = cfg.DATASET.SAMPLES_PATH
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(os.path.join(path, filename)))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
            sample = torch.stack(src)
        latents = rnd.randn(sample.shape[0], cfg.MODEL.LATENT_SPACE_SIZE)
        latents_global = latents if cfg.MODEL.COMMON_Z else rnd.randn(sample.shape[0], cfg.MODEL.LATENT_SPACE_SIZE)
        samplez = torch.tensor(latents).float().cuda()
        samplez_global = torch.tensor(latents_global).float().cuda()
    else:
        dataset_test.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, len(dataset_test.dataset))
        sample_dict_test = next(iter(dataset_test.iterator))
        # sample_dict_test = concate_batch(sample_dict_test)
        sample_spec_test = sample_dict_test['spkr_re_batch_all'].to('cuda').float()
        latents = rnd.randn(sample_spec_test.shape[0], cfg.MODEL.LATENT_SPACE_SIZE)
        latents_global = latents if cfg.MODEL.COMMON_Z else rnd.randn(sample_spec_test.shape[0], cfg.MODEL.LATENT_SPACE_SIZE)
        samplez = torch.tensor(latents).float().cuda()
        samplez_global = torch.tensor(latents_global).float().cuda()
        if cfg.MODEL.ECOG:
            ecog_test = [sample_dict_test['ecog_re_batch_all'][i].to('cuda').float() for i in range(len(sample_dict_test['ecog_re_batch_all']))]
            mask_prior_test = [sample_dict_test['mask_all'][i].to('cuda').float() for i in range(len(sample_dict_test['mask_all']))]
        else:
            ecog_test = None
            mask_prior_test = None
        # sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
        # sample = (sample / 127.5 - 1.)

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

        # batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        for sample_dict_train in tqdm(iter(dataset.iterator)):
            # sample_dict_train = concate_batch(sample_dict_train)
            i += 1
            x_orig = sample_dict_train['spkr_re_batch_all'].to('cuda').float()
            words = sample_dict_train['word_batch_all'].to('cuda').long()
            words = words.view(words.shape[0]*words.shape[1])
            if cfg.MODEL.ECOG:
                ecog = [sample_dict_train['ecog_re_batch_all'][j].to('cuda').float() for j in range(len(sample_dict_train['ecog_re_batch_all']))]
                mask_prior = [sample_dict_train['mask_all'][j].to('cuda').float() for j in range(len(sample_dict_train['mask_all']))]
            else:
                ecog = None
                mask_prior = None
            with torch.no_grad():
                # if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                #     continue
                # if need_permute:
                #     x_orig = x_orig.permute(0, 3, 1, 2)
                # x_orig = (x_orig / 127.5 - 1.)
                x_orig = F.avg_pool2d(x_orig,x_orig.shape[-2]//2**lod2batch.get_lod_power2(),x_orig.shape[-2]//2**lod2batch.get_lod_power2())
                # x_orig = F.interpolate(x_orig, [x_orig.shape[-1]//2**lod2batch.get_lod_power2(),x_orig.shape[-1]//2**lod2batch.get_lod_power2()],mode='bilinear',align_corners=False)
                blend_factor = lod2batch.get_blend_factor()
                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig
                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, scale_factor=2)
                    # x_prev_2x = F.interpolate(x_prev, needed_resolution,mode='bilinear',align_corners=False)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            x.requires_grad = True
            apply_cycle = cfg.MODEL.CYCLE and True
            apply_w_classifier = cfg.MODEL.W_CLASSIFIER and True
            apply_gp = True
            apply_ppl = cfg.MODEL.APPLY_PPL and True
            apply_ppl_d = cfg.MODEL.APPLY_PPL_D and True
            apply_encoder_guide = (cfg.FINETUNE.ENCODER_GUIDE or cfg.MODEL.W_SUP) and True
            apply_sup = cfg.FINETUNE.SPECSUP

            if not (cfg.FINETUNE.FINETUNE):
                encoder_optimizer.zero_grad()
                loss_d = model(x, lod2batch.lod, blend_factor, tracker = tracker, d_train=True, ae=False,words=words,apply_w_classifier=apply_w_classifier, apply_gp = apply_gp,apply_ppl_d=apply_ppl_d,ecog=ecog,mask_prior=mask_prior)
                (loss_d).backward()
                encoder_optimizer.step()

            if cfg.MODEL.ECOG:
                ecog_encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss_g = model(x, lod2batch.lod, blend_factor, tracker = tracker, d_train=False, ae=False,apply_encoder_guide=apply_encoder_guide,apply_ppl=apply_ppl,ecog=ecog,sup=apply_sup,mask_prior=mask_prior,gan=cfg.MODEL.GAN)
            if (cfg.MODEL.ECOG and cfg.MODEL.SUPLOSS_ON_ECOGF) or (cfg.FINETUNE.FINETUNE and cfg.FINETUNE.FIX_GEN ):
                loss_g,loss_sup = loss_g
            # tracker.update(dict(std_scale=model.decoder.std_each_scale))
            if not (cfg.FINETUNE.FINETUNE and cfg.FINETUNE.FIX_GEN):
                (loss_g).backward(retain_graph=True)
                decoder_optimizer.step()
            if (cfg.MODEL.ECOG and cfg.MODEL.SUPLOSS_ON_ECOGF) or (cfg.FINETUNE.FINETUNE and cfg.FINETUNE.FIX_GEN):
                loss_sup.backward()
                ecog_encoder_optimizer.step()

            if not cfg.FINETUNE.FINETUNE:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                lae = model(x, lod2batch.lod, blend_factor, tracker = tracker, d_train=True, ae=True,apply_cycle=apply_cycle,ecog=ecog,mask_prior=mask_prior)
                (lae).backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

            if local_rank == 0:
                betta = 0.5 ** (lod2batch.get_batch_size() / (10 * 1000.0))
                model_s.lerp(model, betta,w_classifier = cfg.MODEL.W_CLASSIFIER)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod_for_saving_model = lod2batch.lod if cfg.TRAIN.PROGRESSIVE else int(epoch//1)
            lod2batch.step()
            if local_rank == 0:
                if lod2batch.is_time_to_save():
                    checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                if lod2batch.is_time_to_report():
                    save_sample(lod2batch, tracker, sample_spec_test, samplez, samplez_global, x, logger, model_s, cfg, encoder_optimizer,
                                decoder_optimizer,ecog=ecog_test,mask_prior=mask_prior_test)
                    if ecog is not None:
                        save_sample(lod2batch, tracker, x_orig, samplez, samplez_global, x, logger, model_s, cfg, encoder_optimizer,
                                    decoder_optimizer,ecog=ecog,mask_prior=mask_prior,mode='train')

        scheduler.step()

        if local_rank == 0:
            checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)
            save_sample(lod2batch, tracker, sample_spec_test, samplez, samplez_global, x, logger, model_s, cfg, encoder_optimizer, decoder_optimizer,
            ecog=ecog_test,mask_prior=mask_prior_test)
            if ecog is not None:
                save_sample(lod2batch, tracker, x_orig, samplez, samplez_global, x, logger, model_s, cfg, encoder_optimizer,
                            decoder_optimizer,ecog=ecog,mask_prior=mask_prior,mode='train')

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/ecog_style2.yaml',
        world_size=gpu_count)
