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
from net import *
from model import Model
from launcher import run

from checkpointer import Checkpointer

from dlutils.pytorch.cuda_helper import *
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from mtcnn.mtcnn import MTCNN
from metrics.inception_resnet_v1 import InceptionResnetV1
from torch.optim.sgd import SGD
from PIL import Image
import bimpy


lreq.use_implicit_lreq.set(True)



class EmbMapping(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(EmbMapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.01)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)
        x = F.normalize(x, p=2, dim=1)
        return x



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

    emb = EmbMapping(mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=1024)
    emb.cuda(0)
    emb_dict = {
        'model': model,
    }
    checkpointer = Checkpointer(cfg, emb_dict,
                                logger=logger,
                                save=False)
    extra_checkpoint_data = checkpointer.load(file_name="embedding/emb_model_final.pth")

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

    path = 'realign1024x1024'
    #path = 'imagenet256x256'
    # path = 'realign128x128'

    paths = list(os.listdir(path))

    paths.sort()

    #random.seed(3456)
    #random.shuffle(paths)
    randomize = bimpy.Bool(True)

    ctx = bimpy.Context()
    v0 = bimpy.Float(0)
    v1 = bimpy.Float(0)
    v2 = bimpy.Float(0)
    v3 = bimpy.Float(0)
    v4 = bimpy.Float(0)
    v10 = bimpy.Float(0)
    v11 = bimpy.Float(0)
    v17 = bimpy.Float(0)
    v19 = bimpy.Float(0)

    w0 = torch.tensor(np.load("direction_%d.npy" % 0), dtype=torch.float32)
    w1 = torch.tensor(np.load("direction_%d.npy" % 1), dtype=torch.float32)
    w2 = torch.tensor(np.load("direction_%d.npy" % 2), dtype=torch.float32)
    w3 = torch.tensor(np.load("direction_%d.npy" % 3), dtype=torch.float32)
    w4 = torch.tensor(np.load("direction_%d.npy" % 4), dtype=torch.float32)
    w10 = torch.tensor(np.load("direction_%d.npy" % 10), dtype=torch.float32)
    w11 = torch.tensor(np.load("direction_%d.npy" % 11), dtype=torch.float32)
    w17 = torch.tensor(np.load("direction_%d.npy" % 17), dtype=torch.float32)
    w19 = torch.tensor(np.load("direction_%d.npy" % 19), dtype=torch.float32)

    _latents = None

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True, device=torch.device('cuda:0')
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def loadNext():
        img = np.asarray(Image.open(path + '/' + paths[0]))
        img_src = img
        paths.pop(0)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]
        _latents = encode(x[None, ...].cuda())
        latents = _latents[0, 0].clone()

        latents -= model.dlatent_avg.buff.data[0]

        v0.value = (latents * w0).sum()
        v1.value = (latents * w1).sum()
        v2.value = (latents * w2).sum()
        v3.value = (latents * w3).sum()
        v4.value = (latents * w4).sum()
        v10.value = (latents * w10).sum()
        v11.value = (latents * w11).sum()
        v17.value = (latents * w17).sum()
        v19.value = (latents * w19).sum()

        latents = latents - v0.value * w0
        latents = latents - v1.value * w1
        latents = latents - v2.value * w2
        latents = latents - v3.value * w3
        latents = latents - v10.value * w10
        latents = latents - v11.value * w11
        latents = latents - v17.value * w17
        latents = latents - v19.value * w19
        return latents, _latents, img_src

    def do_fixup(img, _latents):
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        img_cropped = mtcnn((x[None, ...] * 0.5 + 0.5) * 255)

        latents = _latents[0, 0]

        p = nn.Parameter(data=latents[None, ...].detach())

        optim = SGD([p], lr=200.0)

        e = resnet(img_cropped)

        for i in range(100):
            _p = p[None, ...].repeat(1, model.mapping_fl.num_layers, 1)
            _p = _p.repeat(4, 1, 1)
            x_rec = decode(_p)
            try:
                cropped = mtcnn((x_rec * 0.5 + 0.5) * 255)
                _e = resnet(cropped)
                loss = torch.mean((_e - e.detach()) ** 2)
                print(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
            except:
                break
        latents = p[0, ...]
        _latents = p[None, ...].repeat(1, model.mapping_fl.num_layers, 1)

        latents -= model.dlatent_avg.buff.data[0]

        v0.value = (latents * w0).sum()
        v1.value = (latents * w1).sum()
        v2.value = (latents * w2).sum()
        v3.value = (latents * w3).sum()
        v4.value = (latents * w4).sum()
        v10.value = (latents * w10).sum()
        v11.value = (latents * w11).sum()
        v17.value = (latents * w17).sum()
        v19.value = (latents * w19).sum()

        latents = latents - v0.value * w0
        latents = latents - v1.value * w1
        latents = latents - v2.value * w2
        latents = latents - v3.value * w3
        latents = latents - v10.value * w10
        latents = latents - v11.value * w11
        latents = latents - v17.value * w17
        latents = latents - v19.value * w19
        return latents, _latents

    latents, _latents, img_src = loadNext()

    ctx.init(1800, 1600, "Styles")

    def update_image(w, _w):
        with torch.no_grad():
            w = w + model.dlatent_avg.buff.data[0]
            w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)

            layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = (7 + 1) * 2
            mixing_cutoff = cur_layers
            styles = torch.where(layer_idx < mixing_cutoff, w, _latents[0])

            x_rec = decode(styles)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    im = update_image(latents, _latents)
    print(im.shape)
    im = bimpy.Image(im)

    display_original = True

    seed = 0

    while(not ctx.should_close()):
        with ctx:
            W = latents + w0 * v0.value + w1 * v1.value + w2 * v2.value + w3 * v3.value + w4 * v4.value + w10 * v10.value + w11 * v11.value + w17 * v17.value + w19 * v19.value
            #W = F.leaky_relu(W, 0.2)
            if display_original:
                im = bimpy.Image(img_src)
            else:
                im = bimpy.Image(update_image(W, _latents))

            # if bimpy.button('Ok'):
            bimpy.image(im)
            bimpy.begin("Controls")
            bimpy.slider_float("female <-> male", v0, -30.0, 30.0)
            bimpy.slider_float("smile", v1, -30.0, 30.0)
            bimpy.slider_float("attractive", v2, -30.0, 30.0)
            bimpy.slider_float("wavy-hair", v3, -30.0, 30.0)
            bimpy.slider_float("young", v4, -30.0, 30.0)
            bimpy.slider_float("big lips", v10, -30.0, 30.0)
            bimpy.slider_float("big nose", v11, -30.0, 30.0)
            bimpy.slider_float("chubby", v17, -30.0, 30.0)
            bimpy.slider_float("glasses", v19, -30.0, 30.0)

            bimpy.checkbox("Randomize noise", randomize)

            if randomize.value:
                seed += 1

            torch.manual_seed(seed)

            if bimpy.button('Next'):
                latents, _latents, img_src = loadNext()
                display_original = True

            if bimpy.button('Fix'):
                latents, _latents = do_fixup(img_src, _latents)

            if bimpy.button('Display Reconstruction'):
                display_original = False
            bimpy.end()

    exit()


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='StyleGAN', default_config='configs/experiment_ffhq_z.yaml',
        world_size=gpu_count, write_log=False)
