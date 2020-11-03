import pdb
import random
from tracker import LossTracker
import losses
from net_formant import *
import numpy as np
from torch.nn import functional as F
def compdiff(comp):
    return ((comp[:,:,1:]-comp[:,:,:-1]).abs()).mean()

def compdiffd2(comp):
    diff = comp[:,:,1:]-comp[:,:,:-1]
    return ((diff[:,:,1:]-diff[:,:,:-1]).abs()).mean()

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

class GHMC(nn.Module):
    def __init__(
            self,
            bins=30,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

class GHMR(nn.Module):
    def __init__(
            self,
            mu=0.02,
            bins=30,
            momentum=0,
            loss_weight=1.0):
        super(GHMR, self).__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, avg_factor=None,reweight=1):
        """ Args:
        pred [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of pred.
        label_weight [batch_num, 4 (* class_num)]:
            The weight of each sample, 0 if ignored.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = pred - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = label_weight > 0
        tot = max(label_weight.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = (loss*reweight).sum() / tot
        return loss * self.loss_weight
    
class LAE(nn.Module):
    def __init__(self,mu=0.02,
            bins=30,
            momentum=0.75,
            loss_weight=1.0,db=True,amp=True,noise_db=-50,max_db=22.5):
        super(LAE, self).__init__()
        self.db=db
        self.amp = amp
        self.noise_db = noise_db
        self.max_db = max_db
        if db:
            self.ghm_db = GHMR(mu,bins,momentum,loss_weight)
        if amp:
            self.ghm_amp = GHMR(mu,bins,momentum,loss_weight)

    def forward(self, rec, spec, tracker=None,reweight=1):
        if self.db:
            loss_db = self.ghm_db(rec,spec,torch.ones(spec.shape),reweight=reweight)
            if tracker is not None:
                tracker.update(dict(Lae_db=loss_db))
        else:
            loss_db = torch.tensor(0.0)
        if self.amp:
            spec_amp = amplitude(spec,noise_db=self.noise_db,max_db=self.max_db)
            rec_amp = amplitude(rec,noise_db=self.noise_db,max_db=self.max_db)
            loss_a = self.ghm_amp(rec_amp,spec_amp,torch.ones(spec_amp.shape),reweight=reweight)
            if tracker is not None:
                tracker.update(dict(Lae_a=loss_a))
        else:
            loss_a = torch.tensor(0.0)
        return loss_db+loss_a


class Model(nn.Module):
    def __init__(self, generator="", encoder="", ecog_encoder_name="",
                 spec_chans = 128, n_formants=2, n_formants_noise=2, n_formants_ecog=2, n_fft=256, noise_db=-50, max_db=22.5, wavebased = False,
                 with_ecog = False, ghm_loss=True,power_synth=True,apply_flooding=True,ecog_compute_db_loudness=False,
                 hidden_dim=256,dim_feedforward=256,encoder_only=True,attentional_mask=False,n_heads=1,non_local=False,do_mel_guide = True,noise_from_data=False,specsup=True,\
                 onedconfirst=True,rnn_type = 'LSTM',rnn_layers = 4,compute_db_loudness=True,bidirection = True):
        super(Model, self).__init__()
        self.spec_chans = spec_chans
        self.with_ecog = with_ecog
        self.ecog_encoder_name = ecog_encoder_name
        self.n_formants_ecog = n_formants_ecog
        self.wavebased = wavebased
        self.n_fft = n_fft
        self.n_mels = spec_chans
        self.do_mel_guide = do_mel_guide
        self.noise_db = noise_db
        self.spec_sup = specsup
        self.max_db = max_db
        self.apply_flooding = apply_flooding
        self.n_formants_noise = n_formants_noise
        self.power_synth =power_synth
        self.decoder = GENERATORS[generator](
            n_mels = spec_chans,
            k = 40,
            wavebased = wavebased,
            n_fft = n_fft,
            noise_db = noise_db,
            max_db = max_db,
            noise_from_data = noise_from_data,
            return_wave = False,
            power_synth=power_synth,
        )
        if do_mel_guide:
            self.decoder_mel = GENERATORS[generator](
                n_mels = spec_chans,
                k = 40,
                wavebased = False,
                n_fft = n_fft,
                noise_db = noise_db,
                max_db = max_db,
                add_bgnoise = False,
            )
        self.encoder = ENCODERS[encoder](
            n_mels = spec_chans,
            n_formants = n_formants,
            n_formants_noise = n_formants_noise,
            wavebased = wavebased,
            hop_length = 128,
            n_fft = n_fft,
            noise_db = noise_db,
            max_db = max_db,
            power_synth = power_synth,
        )
        if with_ecog:
            if 'Transformer' in ecog_encoder_name:
                self.ecog_encoder = ECOG_ENCODER[ecog_encoder_name](
                    n_mels = spec_chans,n_formants = n_formants_ecog,
                    hidden_dim=hidden_dim,dim_feedforward=dim_feedforward,n_heads=n_heads,
                    encoder_only=encoder_only,attentional_mask=attentional_mask,non_local=non_local,
                    compute_db_loudness = ecog_compute_db_loudness,
                )
            else:
                self.ecog_encoder = ECOG_ENCODER[ecog_encoder_name](
                    n_mels = spec_chans,n_formants = n_formants_ecog,
                    compute_db_loudness = ecog_compute_db_loudness,
                )
        self.ghm_loss = ghm_loss
        self.lae1 = LAE(noise_db=self.noise_db,max_db=self.max_db)
        self.lae2 = LAE(amp=False)
        self.lae3 = LAE(amp=False)
        self.lae4 = LAE(amp=False)
        self.lae5 = LAE(amp=False)
        self.lae6 = LAE(amp=False)
        self.lae7 = LAE(amp=False) 
        self.lae8 = LAE(amp=False) 

    def noise_dist_init(self,dist):
        with torch.no_grad():
            self.decoder.noise_dist = dist.reshape([1,1,1,dist.shape[0]])

    def generate_fromecog(self, ecog = None, mask_prior = None, mni=None,return_components=False):
        components = self.ecog_encoder(ecog, mask_prior,mni)
        rec = self.decoder.forward(components)
        if return_components:
            return rec, components
        else:
            return rec

    def generate_fromspec(self, spec, return_components=False,x_denoise=None,duomask=False):#,gender='Female'):
        components = self.encoder(spec,x_denoise=x_denoise,duomask=duomask)#,gender=gender)
        rec = self.decoder.forward(components)
        if return_components:
            return rec, components
        else:
            return rec

    def encode(self, spec,x_denoise=None,duomask=False,noise_level = None,x_amp=None):#,gender='Female'):
        components = self.encoder(spec,x_denoise=x_denoise,duomask=duomask,noise_level=noise_level,x_amp=x_amp)#,gender=gender)
        return components

    def lae(self,spec,rec,db=True,amp=True,tracker=None,GHM=False):
        if amp:
            spec_amp = amplitude(spec,noise_db=self.noise_db,max_db=self.max_db)
            rec_amp = amplitude(rec,noise_db=self.noise_db,max_db=self.max_db)
            # if self.power_synth:
            #     spec_amp_ = spec_amp**0.5
            #     rec_amp_ = rec_amp**0.5
            # else:
            #     spec_amp_ = spec_amp
            #     rec_amp_ = rec_amp
            spec_amp_ = spec_amp
            rec_amp_ = rec_amp
            if GHM:
                Lae_a =  self.ghm_loss(rec_amp_,spec_amp_,torch.ones(spec_amp_))#*150
                Lae_a_l2 = torch.tensor([0.])
            else:
                Lae_a =  (spec_amp_-rec_amp_).abs().mean()#*150
                Lae_a_l2 =  torch.sqrt((spec_amp_-rec_amp_)**2+1E-6).mean()#*150
        else:
            Lae_a = torch.tensor(0.)
            Lae_a_l2 = torch.tensor(0.)
        if tracker is not None:
            tracker.update(dict(Lae_a=Lae_a,Lae_a_l2=Lae_a_l2))
        if db:
            if GHM:
                Lae_db =  self.ghm_loss(rec,spec,torch.ones(spec))#*150
                Lae_db_l2 = torch.tensor([0.])
            else:
                Lae_db =  (spec-rec).abs().mean()
                Lae_db_l2 = torch.sqrt((spec-rec)**2+1E-6).mean()
        else:
            Lae_db = torch.tensor(0.)
            Lae_db_l2 = torch.tensor(0.)
        if tracker is not None:
            tracker.update(dict(Lae_db=Lae_db,Lae_db_l2=Lae_db_l2))
        # return (Lae_a + Lae_a_l2)/2. + (Lae_db+Lae_db_l2)/2.
        return Lae_a + Lae_db/2.
    
    def flooding(self, loss,beta):
        if self.apply_flooding:
            return (loss-beta).abs()+beta
        else:
            return loss

    def forward(self, spec, ecog, mask_prior, on_stage, on_stage_wider, ae, tracker, encoder_guide, x_mel=None,x_denoise=None, pitch_aug=False, duomask=False, mni=None,debug=False,x_amp=None,hamonic_bias=False,x_amp_from_denoise=False,gender='Female'):
        if ae:
            self.encoder.requires_grad_(True)
            # rec = self.generate_fromspec(spec)
            components = self.encoder(spec,x_denoise = x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)#,gender=gender)
            rec = self.decoder.forward(components)

            freq_cord = torch.arange(self.spec_chans).reshape([1,1,1,self.spec_chans])/(1.0*self.spec_chans)
            freq_cord2 = torch.arange(self.spec_chans+1).reshape([1,1,1,self.spec_chans+1])/(1.0*self.spec_chans)
            freq_linear_reweighting = 1 if self.wavebased else (inverse_mel_scale(freq_cord2[...,1:])-inverse_mel_scale(freq_cord2[...,:-1]))/440*7
            # freq_linear_reweighting = 1
            # Lae = 4*self.lae((rec*freq_linear_reweighting)[...,128:],(spec*freq_linear_reweighting)[...,128:],tracker=tracker)#torch.mean((rec - spec).abs()*freq_linear_reweighting)
            Lae = 4*self.lae(rec*freq_linear_reweighting,spec*freq_linear_reweighting,tracker=tracker)#torch.mean((rec - spec).abs()*freq_linear_reweighting)
            if self.wavebased:
               spec_amp = amplitude(spec,self.noise_db,self.max_db).transpose(-2,-1)
               rec_amp = amplitude(rec,self.noise_db,self.max_db).transpose(-2,-1)
               freq_cord2 = torch.arange(128+1).reshape([1,1,1,128+1])/(1.0*128)
               freq_linear_reweighting2 = (inverse_mel_scale(freq_cord2[...,1:])-inverse_mel_scale(freq_cord2[...,:-1]))/440*7
               spec_mel = to_db(torchaudio.transforms.MelScale(f_max=8000,n_stft=self.n_fft)(spec_amp).transpose(-2,-1),self.noise_db,self.max_db)
               rec_mel = to_db(torchaudio.transforms.MelScale(f_max=8000,n_stft=self.n_fft)(rec_amp).transpose(-2,-1),self.noise_db,self.max_db)
               Lae += 4*self.lae(rec_mel*freq_linear_reweighting2,spec_mel*freq_linear_reweighting2,tracker=tracker)
               
            #    hann_win = torch.hann_window(21,periodic=False).reshape([1,1,21,1])
            #    spec_broud = to_db(F.conv2d(spec_amp,hann_win,padding=[10,0]).transpose(-2,-1),self.noise_db,self.max_db)
            #    rec_broud = to_db(F.conv2d(rec_amp,hann_win,padding=[10,0]).transpose(-2,-1),self.noise_db,self.max_db)
            #    Lae += 4*self.lae(rec_broud,spec_broud,tracker=tracker)

            if self.do_mel_guide:
               rec_mel = self.decoder_mel.forward(components)
               freq_linear_reweighting_mel = (inverse_mel_scale(freq_cord2[...,1:])-inverse_mel_scale(freq_cord2[...,:-1]))/440*7
               Lae_mel = 4*self.lae(rec_mel*freq_linear_reweighting_mel,x_mel*freq_linear_reweighting_mel,tracker=None)
               tracker.update(dict(Lae_mel=Lae_mel))
               Lae+=Lae_mel

            # rec_denoise = self.decoder.forward(components,enable_hamon_excitation=False,enable_noise_excitation=False)
            # Lae_noise = (50. if self.wavebased else 50.)*self.lae(rec_noise*(1-on_stage_wider.unsqueeze(-1)),spec*(1-on_stage_wider.unsqueeze(-1)))
            # tracker.update(dict(Lae_noise=Lae_noise))
            # Lae += Lae_noise

            if x_amp_from_denoise:
                if self.wavebased:
                    if self.power_synth:
                        Lloudness = 10**6*(components['loudness']*(1-on_stage_wider)).mean()
                    else:
                        # Lloudness = 10**3*(components['loudness']*(1-on_stage_wider)).mean()
                        Lloudness = 10**6*(components['loudness']*(1-on_stage_wider)).mean()
                    # Lloudness = 10.**6*((components['loudness'])**2*(1-on_stage_wider)).mean()
                    tracker.update(dict(Lloudness=Lloudness))
                    Lae += Lloudness

            if self.wavebased and x_denoise is not None:
               thres = int(hz2ind(4000,self.n_fft)) if self.wavebased else mel_scale(self.spec_chans,4000,pt=False).astype(np.int32)
               explosive=(torch.mean((spec*freq_linear_reweighting)[...,thres:],dim=-1)>torch.mean((spec*freq_linear_reweighting)[...,:thres],dim=-1)).to(torch.float32).unsqueeze(-1)
               rec_denoise = self.decoder.forward(components,enable_hamon_excitation=True,enable_noise_excitation=True,enable_bgnoise=False)
               Lae_denoise = 20*self.lae(rec_denoise*freq_linear_reweighting*explosive,x_denoise*freq_linear_reweighting*explosive)
               tracker.update(dict(Lae_denoise=Lae_denoise))
               Lae += Lae_denoise
            # import pdb;pdb.set_trace()
            # if components['freq_formants_hamon'].shape[1] > 2:
            freq_limit = self.encoder.formant_freq_limits_abs.squeeze() 
            from net_formant import mel_scale
            freq_limit = hz2ind(freq_limit,self.n_fft).long() if self.wavebased else mel_scale(self.spec_chans,freq_limit).long()
            if debug:
               import pdb;pdb.set_trace()
            
            
            # if True:
            if not self.wavebased:
                n_formant_noise = components['freq_formants_noise'].shape[1]-components['freq_formants_hamon'].shape[1]
                for formant in range(components['freq_formants_hamon'].shape[1]-1,1,-1):
                     components_copy = {i:j.clone() for i,j in components.items()}
                     components_copy['freq_formants_hamon'] = components_copy['freq_formants_hamon'][:,:formant]
                     components_copy['freq_formants_hamon_hz'] = components_copy['freq_formants_hamon_hz'][:,:formant]
                     components_copy['bandwidth_formants_hamon'] = components_copy['bandwidth_formants_hamon'][:,:formant]
                     components_copy['bandwidth_formants_hamon_hz'] = components_copy['bandwidth_formants_hamon_hz'][:,:formant]
                     components_copy['amplitude_formants_hamon'] = components_copy['amplitude_formants_hamon'][:,:formant]

                     if duomask:
                        # components_copy['freq_formants_noise'] = components_copy['freq_formants_noise'][:,:formant]
                        # components_copy['freq_formants_noise_hz'] = components_copy['freq_formants_noise_hz'][:,:formant]
                        # components_copy['bandwidth_formants_noise'] = components_copy['bandwidth_formants_noise'][:,:formant]
                        # components_copy['bandwidth_formants_noise_hz'] = components_copy['bandwidth_formants_noise_hz'][:,:formant]
                        # components_copy['amplitude_formants_noise'] = components_copy['amplitude_formants_noise'][:,:formant]
                        components_copy['freq_formants_noise'] = torch.cat([components_copy['freq_formants_noise'][:,:formant],components_copy['freq_formants_noise'][:,-n_formant_noise:]],dim=1)
                        components_copy['freq_formants_noise_hz'] = torch.cat([components_copy['freq_formants_noise_hz'][:,:formant],components_copy['freq_formants_noise_hz'][:,-n_formant_noise:]],dim=1)
                        components_copy['bandwidth_formants_noise'] = torch.cat([components_copy['bandwidth_formants_noise'][:,:formant],components_copy['bandwidth_formants_noise'][:,-n_formant_noise:]],dim=1)
                        components_copy['bandwidth_formants_noise_hz'] = torch.cat([components_copy['bandwidth_formants_noise_hz'][:,:formant],components_copy['bandwidth_formants_noise_hz'][:,-n_formant_noise:]],dim=1)
                        components_copy['amplitude_formants_noise'] = torch.cat([components_copy['amplitude_formants_noise'][:,:formant],components_copy['amplitude_formants_noise'][:,-n_formant_noise:]],dim=1)
                  #   rec = self.decoder.forward(components_copy,enable_noise_excitation=True)
                    # Lae += self.lae(rec,spec,tracker=tracker)#torch.mean((rec - spec).abs())
                     rec = self.decoder.forward(components_copy,enable_noise_excitation=True if self.wavebased else True)
                     Lae += 1*self.lae((rec*freq_linear_reweighting),(spec*freq_linear_reweighting),tracker=tracker)#torch.mean(((rec - spec).abs()*freq_linear_reweighting)[...,:freq_limit[formant-1]])
                     #   Lae += self.lae((rec*freq_linear_reweighting)[...,:freq_limit[formant-1]],(spec*freq_linear_reweighting)[...,:freq_limit[formant-1]],tracker=tracker)#torch.mean(((rec - spec).abs()*freq_linear_reweighting)[...,:freq_limit[formant-1]])
                     # Lamp = 1*torch.mean(F.relu(-components['amplitude_formants_hamon'][:,0:3]+components['amplitude_formants_hamon'][:,1:4])*(components['amplitudes'][:,0:1]>components['amplitudes'][:,1:2]).float())
                     # tracker.update(dict(Lamp=Lamp))
                     # Lae+=Lamp
            else:
                Lamp = 10*torch.mean(F.relu(-components['amplitude_formants_hamon'][:,0:3]+components['amplitude_formants_hamon'][:,1:4])*(components['amplitudes'][:,0:1]>components['amplitudes'][:,1:2]).float())
                tracker.update(dict(Lamp=Lamp))
                Lae+=Lamp
            tracker.update(dict(Lae=Lae))
            if debug:
                import pdb;pdb.set_trace()
            
            
            thres = int(hz2ind(4000,self.n_fft)) if self.wavebased else mel_scale(self.spec_chans,4000,pt=False).astype(np.int32)
            explosive=torch.sign(torch.mean((spec*freq_linear_reweighting)[...,thres:],dim=-1)-torch.mean((spec*freq_linear_reweighting)[...,:thres],dim=-1))*0.5+0.5
            Lexp = torch.mean((components['amplitudes'][:,0:1]-components['amplitudes'][:,1:2])*explosive)*100
            tracker.update(dict(Lexp=Lexp))
            Lae += Lexp

            if hamonic_bias:
                hamonic_loss = 1000*torch.mean((1-components['amplitudes'][:,0])*on_stage)
                Lae += hamonic_loss
            
            # alphaloss=(F.relu(0.5-(components['amplitudes']-0.5).abs())*100).mean()
            # Lae+=alphaloss

            if pitch_aug:
               pitch_shift = (2**(-1.5+3*torch.rand([components['f0_hz'].shape[0]]).to(torch.float32)).reshape([components['f0_hz'].shape[0],1,1])) # +- 1 octave
               # pitch_shift = (2**(torch.randint(-1,2,[components['f0_hz'].shape[0]]).to(torch.float32)).reshape([components['f0_hz'].shape[0],1,1])).clamp(min=88,max=616) # +- 1 octave
               components['f0_hz'] = (components['f0_hz']*pitch_shift).clamp(min=88,max=300)
               # components['f0'] = mel_scale(self.spec_chans,components['f0'])/self.spec_chans
               rec_shift = self.decoder.forward(components)
               components_enc = self.encoder(rec_shift,duomask=duomask,x_denoise=x_denoise,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)#,gender=gender)
               Lf0 = torch.mean((components_enc['f0_hz']/200-components['f0_hz']/200)**2)
               rec_cycle = self.decoder.forward(components_enc)
               Lae += self.lae(rec_shift*freq_linear_reweighting,rec_cycle*freq_linear_reweighting,tracker=tracker)#torch.mean((rec_shift-rec_cycle).abs()*freq_linear_reweighting)
               # import pdb;pdb.set_trace()
            else:
               # Lf0 = torch.mean((F.relu(160 - components['f0_hz']) + F.relu(components['f0_hz']-420))/10)
               Lf0 = torch.tensor([0.])
            # Lf0 = torch.tensor([0.])
            tracker.update(dict(Lf0=Lf0))

            spec = spec.squeeze(dim=1).permute(0,2,1) #B * f * T
            loudness = torch.mean(spec*0.5+0.5,dim=1,keepdim=True)
            # import pdb;pdb.set_trace()
            if self.wavebased:
               #  hamonic_components_diff = compdiffd2(components['f0_hz']*2) + compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.  
                if self.power_synth:
                    hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750. + compdiffd2(components['amplitude_formants_hamon'])*1500.+ compdiffd2(components['amplitude_formants_noise'])*1500.#   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50  
                else:
                    # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)  + compdiff(components['amplitudes'])*750. + compdiffd2(components['amplitude_formants_hamon'])*1500.+ compdiffd2(components['amplitude_formants_noise'])*1500.#   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50  
                    hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750. + compdiffd2(components['amplitude_formants_hamon'])*1500.+ compdiffd2(components['amplitude_formants_noise'])*1500.#   + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50  

                # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*2)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.  
               #  hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*1.5) + compdiffd2(components['f0_hz']*8)   + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components['amplitudes'])*750.# + compdiff(components['amplitude_formants_hamon'])*750.  + ((components['loudness']*components['amplitudes'][:,1:]/0.0001)**0.125).mean()*50 + compdiff(components['amplitude_formants_noise'])*750.  
                # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz']*2) + compdiffd2(components['f0_hz']/10) + compdiff(components['amplitude_formants_hamon'])*750. + compdiff(components['amplitude_formants_noise'])*750. + compdiffd2(components['freq_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/10) + compdiff(components['bandwidth_formants_noise_hz'][:,components['freq_formants_hamon_hz'].shape[1]:]/10)
                # hamonic_components_diff = compdiffd2(components['freq_formants_hamon_hz'])+100*compdiffd2(components['f0_hz']*3) + compdiff(components['amplitude_formants_hamon'])*750. + compdiff(components['amplitude_formants_noise'])*750. #+ compdiff(components['freq_formants_noise_hz']*(1-on_stage_wider))
            else:
                hamonic_components_diff = compdiff(components['freq_formants_hamon_hz']*(1-on_stage_wider))+compdiff(components['f0_hz']*(1-on_stage_wider)) + compdiff(components['amplitude_formants_hamon'])*750. + compdiff(components['amplitude_formants_noise'])*750. #+ compdiff(components['freq_formants_noise_hz']*(1-on_stage_wider))
            # hamonic_components_diff = compdiff(components['freq_formants_hamon_hz'])+compdiff(components['f0_hz']) + compdiff(components['amplitude_formants_hamon']*(1-on_stage_wider))*1500. + compdiff(components['amplitude_formants_noise']*(1-on_stage_wider))*1500. + compdiff(components['freq_formants_noise_hz'])
            Ldiff = torch.mean(hamonic_components_diff)/2000.
            # Ldiff = torch.mean(components['freq_formants_hamon'].var()+components['freq_formants_noise'].var())*10
            tracker.update(dict(Ldiff=Ldiff))
            Lae += Ldiff
            # Ldiff = 0
            Lfreqorder = torch.mean(F.relu(components['freq_formants_hamon_hz'][:,:-1]-components['freq_formants_hamon_hz'][:,1:])) #+ (torch.mean(F.relu(components['freq_formants_noise_hz'][:,:-1]-components['freq_formants_noise_hz'][:,1:])) if components['freq_formants_noise_hz'].shape[1]>1 else 0)

            return Lae + Lf0 + Lfreqorder,tracker
        else: #ecog to audio
            self.encoder.requires_grad_(False)
            rec,components_ecog = self.generate_fromecog(ecog,mask_prior,mni=mni,return_components=True)

            ###### mel db flooding
            betas = {'loudness':0.01,'freq_formants_hamon':0.0025,'f0_hz':0.,'amplitudes':0.,'amplitude_formants_hamon':0.,'amplitude_formants_noise':0.,'freq_formants_noise':0.05,'bandwidth_formants_noise_hz':0.01}
            alpha = {'loudness':1.,'freq_formants_hamon':4.,'f0_hz':1.,'amplitudes':1.,'amplitude_formants_hamon':1.,'amplitude_formants_noise':1.,'freq_formants_noise':1.,'bandwidth_formants_noise_hz':1.}
            if self.spec_sup:
                if False:#self.ghm_loss:
                    Lrec = 0.3*self.lae1(rec,spec,tracker=tracker)
                else:
                    Lrec = self.lae(rec,spec,tracker=tracker)#torch.mean((rec - spec)**2)
                # Lamp = 10*torch.mean(F.relu(-components_ecog['amplitude_formants_hamon'][:,0:min(3,self.n_formants_ecog-1)]+components_ecog['amplitude_formants_hamon'][:,1:min(4,self.n_formants_ecog)])*(components_ecog['amplitudes'][:,0:1]>components_ecog['amplitudes'][:,1:2]).float())
                # tracker.update(dict(Lamp=Lamp))
                # Lrec+=Lamp
            else:
                Lrec = torch.tensor([0.0])#
            # Lrec = torch.mean((rec - spec).abs())
            tracker.update(dict(Lrec=Lrec))
            Lcomp = 0
            if encoder_guide:
                components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)#,gender=gender)   
                consonant_weight = 1#100*(torch.sign(components_guide['amplitudes'][:,1:]-0.5)*0.5+0.5)
                if self.power_synth:
                    loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
                    loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
                else:
                    loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
                    # loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
                    loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
                   # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
                #loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
                for key in ['loudness','f0_hz','amplitudes','amplitude_formants_hamon','freq_formants_hamon','amplitude_formants_noise','freq_formants_noise','bandwidth_formants_noise_hz']:
                    # if 'hz' in key:
                    #     continue
                    if key == 'loudness':
                        if self.power_synth:
                            loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
                        else:
                            loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
                           # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+35)/25
                            # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key]**2)+70)/50
                        if False:#self.ghm_loss:
                            diff = self.lae2(loudness_db_norm, loudness_db_norm_ecog)
                        else:
                            diff = alpha['loudness']*15*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2)#+ torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
                        diff = self.flooding(diff,alpha['loudness']*betas['loudness'])
                        tracker.update({'loudness_metric' : torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2*on_stage_wider)})

                    if key == 'f0_hz':
                        # diff = torch.mean((components_guide[key]*6 - components_ecog[key]*6)**2 * on_stage_wider * components_guide['loudness']/4)
                        diff = alpha['f0_hz']*0.3*torch.mean((components_guide[key]/200*5 - components_ecog[key]/200*5)**2 * on_stage_wider * loudness_db_norm)
                        diff = self.flooding(diff,alpha['f0_hz']*betas['f0_hz'])
                        tracker.update({'f0_metric' : torch.mean((components_guide['f0_hz']/200*5 - components_ecog['f0_hz']/200*5)**2 * on_stage_wider * loudness_db_norm)})

                    if key in ['amplitudes']:
                    # if key in ['amplitudes','amplitudes_h']:
                        weight = on_stage_wider  * loudness_db_norm
                        if self.ghm_loss:
                            # diff = 100*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
                            diff = alpha['amplitudes']*30*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
                        else:
                            diff = alpha['amplitudes']*10*torch.mean((components_guide[key] - components_ecog[key])**2 *weight)
                        diff = self.flooding(diff,alpha['amplitudes']*betas['amplitudes'])
                        tracker.update({'amplitudes_metric' : torch.mean((components_guide['amplitudes'] - components_ecog['amplitudes'])**2 *weight)})

                    if key in ['amplitude_formants_hamon']:
                        weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
                        if False:#self.ghm_loss:
                            diff = 40*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
                            # diff = 10*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
                        else:
                            # diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
                            # diff = 40*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)/2 \
                            #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(components_guide[key][:,:self.n_formants_ecog])/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
                            diff = alpha['amplitude_formants_hamon']*40*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
                            # diff = 10*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
                        # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
                    # if key in ['freq_formants_hamon']:
                    #     diff = torch.mean((components_guide[key][:,:1]*10 - components_ecog[key][:,:1]*10)**2 * components_guide['amplitude_formants_hamon'][:,:1] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm )
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                        diff = self.flooding(diff,alpha['amplitude_formants_hamon']*betas['amplitude_formants_hamon'])
                        tracker.update({'amplitude_formants_hamon_metric' : torch.mean((components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] - components_ecog['amplitude_formants_hamon'])**2 *  weight)})

                    # if key in ['freq_formants_hamon_hz']:
                    #     # weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
                    #     weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
                    #     if False:#self.ghm_loss:
                    #         diff = 50*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
                    #         # diff = 15*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
                    #     else:
                    #         # diff = 300*torch.mean((components_guide['freq_formants_hamon'][:,:2] - components_ecog['freq_formants_hamon'][:,:2])**2 * weight)
                    #         # diff = 300*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                    #         diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                    #         # diff = 30*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    
                    if key in ['freq_formants_hamon']:
                        weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
                        # weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
                        if False:#self.ghm_loss:
                            diff = 50*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
                            # diff = 15*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
                        else:
                            diff = alpha['freq_formants_hamon']*300*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key][:,:self.n_formants_ecog])**2 * weight)
                            # diff = 300*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                            # diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                            # diff = 30*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
                        # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
                        # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    # if key in ['bandwidth_formants_hamon']:
                    #     diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                        diff = self.flooding(diff,alpha['freq_formants_hamon']*betas['freq_formants_hamon'])
                        tracker.update({'freq_formants_hamon_hz_metric_2' : torch.mean((components_guide['freq_formants_hamon_hz'][:,:2]/400 - components_ecog['freq_formants_hamon_hz'][:,:2]/400)**2 * weight)})
                        tracker.update({','+str(self.n_formants_ecog) : torch.mean((components_guide['freq_formants_hamon_hz'][:,:self.n_formants_ecog]/400 - components_ecog['freq_formants_hamon_hz'][:,:self.n_formants_ecog]/400)**2 * weight)})

                    if key in ['amplitude_formants_noise']:
                        weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight * loudness_db_norm
                        if False:#self.ghm_loss:
                            diff = self.lae6(components_guide[key],components_ecog[key],reweight=weight)
                        else:
                            # diff = 40*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)/2 \
                            #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1))/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
                            diff = alpha['amplitude_formants_noise']*40*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)
                        diff = self.flooding(diff,alpha['amplitude_formants_noise']*betas['amplitude_formants_noise'])
                        tracker.update({'amplitude_formants_noise_metric': torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)})

                    if key in ['freq_formants_noise']:
                        weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
                        if False:#self.ghm_loss:
                            diff = 10*self.lae7(components_guide[key][:,-self.n_formants_noise:]/400,components_ecog[key][:,-self.n_formants_noise:]/400,reweight=weight)
                        else:
                            
                            # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                            diff = alpha['freq_formants_noise']*12000*torch.mean((components_guide[key][:,-self.n_formants_noise:] - components_ecog[key][:,-self.n_formants_noise:])**2 * weight)
                        # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_noise'][:,:self.n_formants_ecog] * components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight)
                        diff = self.flooding(diff,alpha['freq_formants_noise']*betas['freq_formants_noise'])
                        tracker.update({'freq_formants_noise_metic': torch.mean((components_guide['freq_formants_noise_hz'][:,-self.n_formants_noise:]/2000*5 - components_ecog['freq_formants_noise_hz'][:,-self.n_formants_noise:]/2000*5)**2 * weight)})

                    # if key in ['freq_formants_noise_hz']:
                    #     weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
                    #     if False:#self.ghm_loss:
                    #         diff = 10*self.lae7(components_guide[key][:,-self.n_formants_noise:]/400,components_ecog[key][:,-self.n_formants_noise:]/400,reweight=weight)
                    #     else:
                            
                    #         # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                    #         diff = 3*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                    #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_noise'][:,:self.n_formants_ecog] * components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight)
                    #     diff = self.flooding(diff,betas['freq_formants_noise_hz'])
                    #     tracker.update({'freq_formants_noise_hz_metic': torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)})

                    if key in ['bandwidth_formants_noise_hz']:
                        weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
                        if False:#self.ghm_loss:
                            diff = 3*self.lae8(components_guide[key][:,-self.n_formants_noise:]/2000*5, components_ecog[key][:,-self.n_formants_noise:]/2000*5,reweight=weight)
                        else:
                            # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                            diff = alpha['bandwidth_formants_noise_hz']*3*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                        diff = self.flooding(diff,alpha['bandwidth_formants_noise_hz']*betas['bandwidth_formants_noise_hz'])
                        tracker.update({'bandwidth_formants_noise_hz_metic': torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)})
                    
                    # if key in ['bandwidth_formants_noise']:
                    #     weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
                    #     if False:#self.ghm_loss:
                    #         diff = 3*self.lae8(components_guide[key][:,-self.n_formants_noise:], components_ecog[key][:,-self.n_formants_noise:],reweight=weight)
                    #     else:
                    #         # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:] - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                    #         diff = 300*torch.mean((components_guide[key][:,-self.n_formants_noise:] - components_ecog[key][:,-self.n_formants_noise:])**2 * weight)
                    #     diff = self.flooding(diff,betas['bandwidth_formants_noise'])
                    #     tracker.update({'bandwidth_formants_noise_metic': torch.mean((components_guide['bandwidth_formants_noise_hz'][:,-self.n_formants_noise:]/2000*5 - components_ecog['bandwidth_formants_noise_hz'][:,-self.n_formants_noise:]/2000*5)**2 * weight)})
                    tracker.update({key : diff})
                    Lcomp += diff
                    # import pdb; pdb.set_trace()
            
            Loss = Lrec+Lcomp

            hamonic_components_diff = compdiffd2(components_ecog['freq_formants_hamon_hz']*1.5) + compdiffd2(components_ecog['f0_hz']*2)   + compdiff(components_ecog['bandwidth_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components_ecog['freq_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components_ecog['amplitudes'])*750.
            Ldiff = torch.mean(hamonic_components_diff)/2000.
            tracker.update(dict(Ldiff=Ldiff))
            Loss += Ldiff

            freq_linear_reweighting = 1
            thres = int(hz2ind(4000,self.n_fft)) if self.wavebased else mel_scale(self.spec_chans,4000,pt=False).astype(np.int32)
            explosive=torch.sign(torch.mean((spec*freq_linear_reweighting)[...,thres:],dim=-1)-torch.mean((spec*freq_linear_reweighting)[...,:thres],dim=-1))*0.5+0.5
            Lexp = torch.mean((components_ecog['amplitudes'][:,0:1]-components_ecog['amplitudes'][:,1:2])*explosive)*100
            tracker.update(dict(Lexp=Lexp))
            Loss += Lexp

            Lfreqorder = torch.mean(F.relu(components_ecog['freq_formants_hamon_hz'][:,:-1]-components_ecog['freq_formants_hamon_hz'][:,1:]))
            Loss += Lfreqorder

            return Loss,tracker
    # '''
    #         loss_weights_dict = {'Lrec':1,'loudness':15,'f0_hz':0.3,'amplitudes':30,\
    #             'amplitude_formants_hamon':40, 'freq_formants_hamon_hz':200,'amplitude_formants_noise':40,\
    #                 'freq_formants_noise_hz':3,'bandwidth_formants_noise_hz':1,'Ldiff':1/2000.,\
    #                   'Lexp':100, 'Lfreqorder':1  }
    #         self.encoder.requires_grad_(False)
    #         rec,components_ecog = self.generate_fromecog(ecog,mask_prior,mni=mni,return_components=True)

    #         ######
    #         if self.spec_sup:
    #             if False:#self.ghm_loss:
    #                 Lrec = 0.3*self.lae1(rec,spec,tracker=tracker)
    #             else:
    #                 Lrec = loss_weights_dict['Lrec']*self.lae(rec,spec,tracker=tracker)#torch.mean((rec - spec)**2)
    #             # Lamp = 10*torch.mean(F.relu(-components_ecog['amplitude_formants_hamon'][:,0:min(3,self.n_formants_ecog-1)]+components_ecog['amplitude_formants_hamon'][:,1:min(4,self.n_formants_ecog)])*(components_ecog['amplitudes'][:,0:1]>components_ecog['amplitudes'][:,1:2]).float())
    #             # tracker.update(dict(Lamp=Lamp))
    #             # Lrec+=Lamp
    #         else:
    #             Lrec = torch.tensor([0.0])#
    #         # Lrec = torch.mean((rec - spec).abs())
    #         tracker.update(dict(Lrec=Lrec))
    #         Lcomp = 0
    #         if encoder_guide:
    #             components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)   
    #             consonant_weight = 1#100*(torch.sign(components_guide['amplitudes'][:,1:]-0.5)*0.5+0.5)
    #             if self.power_synth:
    #                 loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
    #                 loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
    #             else:
    #                 loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
    #                 # loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
    #                 loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
    #                # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
    #             #loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
    #             for key in ['loudness','f0_hz','amplitudes','amplitude_formants_hamon','freq_formants_hamon_hz','amplitude_formants_noise','freq_formants_noise_hz','bandwidth_formants_noise']:
    #                 # if 'hz' in key:
    #                 #     continue
    #                 #'''
    #                 if key == 'loudness':
    #                     if self.power_synth:
    #                         loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
    #                     else:
    #                         loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
    #                        # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+35)/25
    #                         # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key]**2)+70)/50
    #                     if False:#self.ghm_loss:
    #                         diff = self.lae2(loudness_db_norm, loudness_db_norm_ecog)
    #                     else:
    #                         diff = loss_weights_dict[key]*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2)#+ torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
    #                 #'''
    #                 if key == 'loudness':
    #                     if self.power_synth:
    #                         loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
    #                     else:
    #                         loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
    #                        # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+35)/25
    #                         # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key]**2)+70)/50
    #                     if False:#self.ghm_loss:
    #                         diff = self.lae2(loudness_db_norm, loudness_db_norm_ecog)
    #                     else:
    #                         diff = alpha['loudness']*loss_weights_dict[key]*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2)#+ torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
    #                     diff = self.flooding(diff,alpha['loudness']*betas['loudness'])
    #                     tracker.update({'loudness_metric' : torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2*on_stage_wider)})
                    
    #                 if key == 'f0_hz':
    #                     # diff = torch.mean((components_guide[key]*6 - components_ecog[key]*6)**2 * on_stage_wider * components_guide['loudness']/4)
    #                     diff = loss_weights_dict[key]*torch.mean((components_guide[key]/200*5 - components_ecog[key]/200*5)**2 * on_stage_wider * loudness_db_norm)
    #                 if key in ['amplitudes']:
    #                 # if key in ['amplitudes','amplitudes_h']:
    #                     weight = on_stage_wider  * loudness_db_norm
    #                     if self.ghm_loss:
    #                         # diff = 100*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
    #                         diff = loss_weights_dict[key]*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
    #                     else:
    #                         diff = 10*torch.mean((components_guide[key] - components_ecog[key])**2 *weight)
    #                 if key in ['amplitude_formants_hamon']:
    #                     weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
    #                     if False:#self.ghm_loss:
    #                         diff = 40*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
    #                         # diff = 10*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
    #                     else:
    #                         # diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
    #                         # diff = 40*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)/2 \
    #                         #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(components_guide[key][:,:self.n_formants_ecog])/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
    #                         diff = loss_weights_dict[key]*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
    #                         # diff = 10*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
    #                     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
    #                 # if key in ['freq_formants_hamon']:
    #                 #     diff = torch.mean((components_guide[key][:,:1]*10 - components_ecog[key][:,:1]*10)**2 * components_guide['amplitude_formants_hamon'][:,:1] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm )
    #                 #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
    #                 #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    
    #                 if key in ['freq_formants_hamon_hz']:
    #                     # weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
    #                     weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
    #                     if False:#self.ghm_loss:
    #                         diff = 50*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
    #                         # diff = 15*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
    #                     else:
    #                         # diff = 300*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
    #                         diff = loss_weights_dict[key]*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
    #                         # diff = 30*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
    #                     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
    #                     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
    #                 # if key in ['bandwidth_formants_hamon']:
    #                 #     diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
    #                 #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
    #                 #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    
    #                 if key in ['amplitude_formants_noise']:
    #                     weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight * loudness_db_norm
    #                     if False:#self.ghm_loss:
    #                         diff = self.lae6(components_guide[key],components_ecog[key],reweight=weight)
    #                     else:
    #                         # diff = 40*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)/2 \
    #                         #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1))/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
    #                         diff = loss_weights_dict[key]*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)

    #                 if key in ['freq_formants_noise_hz']:
    #                     weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
    #                     if False:#self.ghm_loss:
    #                         diff = 10*self.lae7(components_guide[key][:,-self.n_formants_noise:]/400,components_ecog[key][:,-self.n_formants_noise:]/400,reweight=weight)
    #                     else:
    #                         # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
    #                         diff = loss_weights_dict[key]*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
    #                     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_noise'][:,:self.n_formants_ecog] * components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight)
    #                 if key in ['bandwidth_formants_noise_hz']:
    #                     weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
    #                     if False:#self.ghm_loss:
    #                         diff = 3*self.lae8(components_guide[key][:,-self.n_formants_noise:]/2000*5, components_ecog[key][:,-self.n_formants_noise:]/2000*5,reweight=weight)
    #                     else:
    #                         # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
    #                         diff = loss_weights_dict[key]*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
    #                 tracker.update({key : diff})
    #                 Lcomp += diff
    #                 # import pdb; pdb.set_trace()
            
    #         Loss = Lrec+Lcomp

    #         hamonic_components_diff = compdiffd2(components_ecog['freq_formants_hamon_hz']*1.5) + compdiffd2(components_ecog['f0_hz']*2)   + compdiff(components_ecog['bandwidth_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components_ecog['freq_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components_ecog['amplitudes'])*750.
    #         Ldiff = loss_weights_dict['Ldiff']*torch.mean(hamonic_components_diff)
    #         tracker.update(dict(Ldiff=Ldiff))
    #         Loss += Ldiff

    #         freq_linear_reweighting = 1
    #         thres = int(hz2ind(4000,self.n_fft)) if self.wavebased else mel_scale(self.spec_chans,4000,pt=False).astype(np.int32)
    #         explosive=torch.sign(torch.mean((spec*freq_linear_reweighting)[...,thres:],dim=-1)-torch.mean((spec*freq_linear_reweighting)[...,:thres],dim=-1))*0.5+0.5
    #         Lexp = loss_weights_dict['Lexp']*torch.mean((components_ecog['amplitudes'][:,0:1]-components_ecog['amplitudes'][:,1:2])*explosive)
    #         tracker.update(dict(Lexp=Lexp))
    #         Loss += Lexp

    #         Lfreqorder = loss_weights_dict['Lfreqorder']*torch.mean(F.relu(components_ecog['freq_formants_hamon_hz'][:,:-1]-components_ecog['freq_formants_hamon_hz'][:,1:]))
    #         Loss += Lfreqorder
    #         #print ('tracker content:',tracker,tracker.keys())
    #         return Loss,tracker
    # '''

            ######### new balanced loss
            # if self.spec_sup:
            #     if False:#self.ghm_loss:
            #         Lrec = 0.3*self.lae1(rec,spec,tracker=tracker)
            #     else:
            #         Lrec = self.lae(rec,spec,tracker=tracker)#torch.mean((rec - spec)**2)
            #     # Lamp = 10*torch.mean(F.relu(-components_ecog['amplitude_formants_hamon'][:,0:min(3,self.n_formants_ecog-1)]+components_ecog['amplitude_formants_hamon'][:,1:min(4,self.n_formants_ecog)])*(components_ecog['amplitudes'][:,0:1]>components_ecog['amplitudes'][:,1:2]).float())
            #     # tracker.update(dict(Lamp=Lamp))
            #     # Lrec+=Lamp
            # else:
            #     Lrec = torch.tensor([0.0])#
            # # Lrec = torch.mean((rec - spec).abs())
            # tracker.update(dict(Lrec=Lrec))
            # Lcomp = 0
            # if encoder_guide:
            #     components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)   
            #     consonant_weight = 1#100*(torch.sign(components_guide['amplitudes'][:,1:]-0.5)*0.5+0.5)
            #     if self.power_synth:
            #         loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #         loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            #     else:
            #         loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #         # loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
            #         loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            #         # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
            #     for key in ['loudness','f0_hz','amplitudes','amplitude_formants_hamon','freq_formants_hamon_hz','amplitude_formants_noise','freq_formants_noise_hz','bandwidth_formants_noise_hz']:
            #         # if 'hz' in key:
            #         #     continue
            #         if key == 'loudness':
            #             if self.power_synth:
            #                 loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
            #             else:
            #                 # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+35)/25
            #                 loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog[key])+70)/50
            #             if False:#self.ghm_loss:
            #                 diff = self.lae2(loudness_db_norm, loudness_db_norm_ecog)
            #             else:
            #                 diff = 10*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2*on_stage_wider) + 1.5*10**6*torch.mean((components_guide['loudness'] - components_ecog['loudness'])**2*on_stage_wider) + 2*10**7*(components_ecog['loudness']**2*(1-on_stage_wider)).mean()#+ torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
            #         if key == 'f0_hz':
            #             # diff = torch.mean((components_guide[key]*6 - components_ecog[key]*6)**2 * on_stage_wider * components_guide['loudness']/4)
            #             diff = torch.mean((components_guide[key]/200*5 - components_ecog[key]/200*5)**2 * on_stage_wider * loudness_db_norm)
            #         if key in ['amplitudes']:
            #         # if key in ['amplitudes','amplitudes_h']:
            #             weight = on_stage_wider  * loudness_db_norm
            #             if self.ghm_loss:
            #                 # diff = 100*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
            #                 diff = 30*self.lae3(components_guide[key], components_ecog[key],reweight=weight)
            #             else:
            #                 diff = 10*torch.mean((components_guide[key] - components_ecog[key])**2 *weight)
            #         if key in ['amplitude_formants_hamon']:
            #             weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            #             if False:#self.ghm_loss:
            #                 diff = 40*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
            #                 # diff = 10*self.lae4(components_guide[key][:,:self.n_formants_ecog], components_ecog[key],reweight=weight)
            #             else:
            #                 # diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
            #                 # diff = 40*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)/2 \
            #                 #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(components_guide[key][:,:self.n_formants_ecog])/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
            #                 diff = 20*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
            #                 # diff = 10*torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 *  weight)
            #             # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
            #         # if key in ['freq_formants_hamon']:
            #         #     diff = torch.mean((components_guide[key][:,:1]*10 - components_ecog[key][:,:1]*10)**2 * components_guide['amplitude_formants_hamon'][:,:1] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm )
            #         #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
            #         #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    
            #         if key in ['freq_formants_hamon_hz']:
            #             # weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            #             weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            #             if False:#self.ghm_loss:
            #                 diff = 50*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
            #                 # diff = 15*self.lae5(components_guide[key][:,:self.n_formants_ecog]/400 , components_ecog[key]/400, reweight=weight)
            #             else:
            #                 diff = 150*torch.mean((components_guide['freq_formants_hamon'][:,:self.n_formants_ecog] - components_ecog['freq_formants_hamon'][:,:self.n_formants_ecog])**2 * weight) \
            #                      + 5*torch.mean((components_guide['freq_formants_hamon_hz'][:,:self.n_formants_ecog]/400 - components_ecog['freq_formants_hamon_hz'][:,:self.n_formants_ecog]/400)**2 * weight)
            #                 # diff = 300*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
            #                 # diff = 100*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
            #                 # diff = 30*torch.mean((components_guide[key][:,:self.n_formants_ecog]/2000*5 - components_ecog[key][:,:self.n_formants_ecog]/2000*5)**2 * weight)
            #             # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]*10 - components_ecog[key]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
            #             # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
            #         # if key in ['bandwidth_formants_hamon']:
            #         #     diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
            #         #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog]/4 - components_ecog[key]/4)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
            #         #     # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
                    
            #         if key in ['amplitude_formants_noise']:
            #             weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight * loudness_db_norm
            #             if False:#self.ghm_loss:
            #                 diff = self.lae6(components_guide[key],components_ecog[key],reweight=weight)
            #             else:
            #                 # diff = 40*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)/2 \
            #                 #      + 40*torch.mean((torchaudio.transforms.AmplitudeToDB()(torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1))/100 - torchaudio.transforms.AmplitudeToDB()(components_ecog[key])/100)**2 *  weight)/2
            #                 diff = 40*torch.mean((torch.cat([components_guide[key][:,:self.n_formants_ecog],components_guide[key][:,-self.n_formants_noise:]],dim=1) - components_ecog[key])**2 *weight)

            #         if key in ['freq_formants_noise_hz']:
            #             weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
            #             if False:#self.ghm_loss:
            #                 diff = 10*self.lae7(components_guide[key][:,-self.n_formants_noise:]/400,components_ecog[key][:,-self.n_formants_noise:]/400,reweight=weight)
            #             else:
            #                 # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
            #                 diff = 1.5*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
            #             # diff = torch.mean((components_guide[key][:,:self.n_formants_ecog] - components_ecog[key])**2 * components_guide['amplitude_formants_noise'][:,:self.n_formants_ecog] * components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight)
            #         if key in ['bandwidth_formants_noise_hz']:
            #             weight = components_guide['amplitudes'][:,1:2] * on_stage_wider * consonant_weight* loudness_db_norm
            #             if False:#self.ghm_loss:
            #                 diff = 3*self.lae8(components_guide[key][:,-self.n_formants_noise:]/2000*5, components_ecog[key][:,-self.n_formants_noise:]/2000*5,reweight=weight)
            #             else:
            #                 # diff = 30*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
            #                 diff = 4*torch.mean((components_guide[key][:,-self.n_formants_noise:]/2000*5 - components_ecog[key][:,-self.n_formants_noise:]/2000*5)**2 * weight)
                    
            #         if key in ['loudness','freq_formants_hamon_hz']:
            #             diff = diff*10.
            #         tracker.update({key : diff})
            #         Lcomp += diff
            # Lcomp = Lcomp/20.
            # Loss = Lrec+Lcomp

            # hamonic_components_diff = compdiffd2(components_ecog['freq_formants_hamon_hz']*1.5) + compdiffd2(components_ecog['f0_hz']*2)   + compdiff(components_ecog['bandwidth_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)  + compdiff(components_ecog['freq_formants_noise_hz'][:,components_ecog['freq_formants_hamon_hz'].shape[1]:]/5)+ compdiff(components_ecog['amplitudes'])*750.
            # Ldiff = torch.mean(hamonic_components_diff)/2000.
            # tracker.update(dict(Ldiff=Ldiff))
            # Loss += Ldiff

            # freq_linear_reweighting = 1
            # thres = int(hz2ind(4000,self.n_fft)) if self.wavebased else mel_scale(self.spec_chans,4000,pt=False).astype(np.int32)
            # explosive=torch.sign(torch.mean((spec*freq_linear_reweighting)[...,thres:],dim=-1)-torch.mean((spec*freq_linear_reweighting)[...,:thres],dim=-1))*0.5+0.5
            # Lexp = torch.mean((components_ecog['amplitudes'][:,0:1]-components_ecog['amplitudes'][:,1:2])*explosive)*100
            # tracker.update(dict(Lexp=Lexp))
            # Loss += Lexp

            # Lfreqorder = torch.mean(F.relu(components_ecog['freq_formants_hamon_hz'][:,:-1]-components_ecog['freq_formants_hamon_hz'][:,1:]))
            # Loss += Lfreqorder

            # return Loss

            ##### fit f1 freq only
            # components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)
            # if self.power_synth:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # else:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     #loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            #     # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
            # # loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # weight = components_guide['amplitudes'][:,0:1] * on_stage_wider * 1 * loudness_db_norm
            # # weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            # if False:#self.ghm_loss:
            #     diff = 50*self.lae5(components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog]/400 , components_ecog[freq_formants_hamon_hz]/400, reweight=weight)
            #     # diff = 15*self.lae5(components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog]/400 , components_ecog[freq_formants_hamon_hz]/400, reweight=weight)
            # else:
            #     diff = 75*torch.mean((components_guide['freq_formants_hamon'][:,:2] - components_ecog['freq_formants_hamon'][:,:2])**2 * weight) \
            #     + 0.5*torch.mean((components_guide['freq_formants_hamon_hz'][:,:2]/400 - components_ecog['freq_formants_hamon_hz'][:,:2]/400)**2 * weight)
            #     # diff = 300*torch.mean((components_guide['freq_formants_hamon_hz'][:,:2]/2000*5 - components_ecog['freq_formants_hamon_hz'][:,:2]/2000*5)**2 * weight)
            #     # diff = 100*torch.mean((components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog]/2000*5 - components_ecog[freq_formants_hamon_hz][:,:self.n_formants_ecog]/2000*5)**2 * weight)
            #     # diff = 30*torch.mean((components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog]/2000*5 - components_ecog[freq_formants_hamon_hz][:,:self.n_formants_ecog]/2000*5)**2 * weight)
            # # diff = torch.mean((components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog]*10 - components_ecog[freq_formants_hamon_hz]*10)**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * components_guide['loudness']/4 * consonant_weight)
            # # diff = torch.mean((components_guide[freq_formants_hamon_hz][:,:self.n_formants_ecog] - components_ecog[freq_formants_hamon_hz])**2 * components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_ecog['amplitudes'][:,0:1] * on_stage_wider * consonant_weight)
            # Loss = diff
            # return Loss

            #### fit loudness freq only
            # components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)
            # if self.power_synth:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # else:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     #loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            #     # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
            # # loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # if self.power_synth:
            #     loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+70)/50
            # else:
            #     # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+35)/25
            #     loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+70)/50
            #     # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness']**2)+70)/50
            # weight = components_guide['loudness'][:,0:1] * on_stage_wider * 1 * loudness_db_norm
            # # weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            # diff = 3*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2) + 10**6*torch.mean((components_guide['loudness'] - components_ecog['loudness'])**2)
            # Loss = diff

            # Lloudness = 10**7*(components_ecog['loudness']**2*(1-on_stage_wider)).mean()
            # Loss+=Lloudness
            # return Loss

            #### fit f1f2 freq and loudness
            # components_guide = self.encode(spec,x_denoise=x_denoise,duomask=duomask,noise_level = F.softplus(self.decoder.bgnoise_amp)*self.decoder.noise_dist.mean(),x_amp=x_amp)
            # if self.power_synth:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # else:
            #     loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness'])
            #     #loudness_db_norm = (loudness_db.clamp(min=-35)+35)/25
            #     loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            #     # loudness_db = torchaudio.transforms.AmplitudeToDB()(components_guide['loudness']**2)
            # # loudness_db_norm = (loudness_db.clamp(min=-70)+70)/50
            # if self.power_synth:
            #     loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+70)/50
            # else:
            #     # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+35)/25
            #     loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness'])+70)/50
            #     # loudness_db_norm_ecog = (torchaudio.transforms.AmplitudeToDB()(components_ecog['loudness']**2)+70)/50
            # weight = components_guide['loudness'][:,0:1] * on_stage_wider * 1 * loudness_db_norm
            # # weight = components_guide['amplitude_formants_hamon'][:,:self.n_formants_ecog] *  components_guide['amplitudes'][:,0:1] * on_stage_wider * consonant_weight * loudness_db_norm
            # diff = 3*torch.mean((loudness_db_norm - loudness_db_norm_ecog)**2) + 10**6*torch.mean((components_guide['loudness'] - components_ecog['loudness'])**2)
            # Loss = diff
            
            # Lloudness = 10**7*(components_ecog['loudness']**2*(1-on_stage_wider)).mean()
            # Loss+=Lloudness

            # weight = on_stage_wider * 1 * loudness_db_norm
            # diff = 75*torch.mean((components_guide['freq_formants_hamon'][:,:2] - components_ecog['freq_formants_hamon'][:,:2])**2 * weight) \
            #      + 0.5*torch.mean((components_guide['freq_formants_hamon_hz'][:,:2]/400 - components_ecog['freq_formants_hamon_hz'][:,:2]/400)**2 * weight)
            # Loss+=diff
            # return Loss

    def lerp(self, other, betta,w_classifier=False):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.decoder.parameters()) + list(self.encoder.parameters()) + (list(self.ecog_encoder.parameters()) if self.with_ecog else []) + (list(self.decoder_mel.parameters()) if self.do_mel_guide else [])
            other_param = list(other.decoder.parameters()) + list(other.encoder.parameters()) + (list(other.ecog_encoder.parameters()) if self.with_ecog else []) + (list(self.decoder_mel.parameters()) if self.do_mel_guide else [])
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
# 