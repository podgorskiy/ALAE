import os
import pdb
from random import triangular
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import lreq as ln
import math
from registry import *
from transformer_models.position_encoding import build_position_encoding
from transformer_models.transformer import Transformer as TransformerTS
from transformer_models.transformer_nonlocal import Transformer as TransformerNL

def db(x,noise = -80, slope =35, powerdb=True):
   if powerdb:
      return ((2*torchaudio.transforms.AmplitudeToDB()(x)).clamp(min=noise)-slope-noise)/slope
   else:
      return ((torchaudio.transforms.AmplitudeToDB()(x)).clamp(min=noise)-slope-noise)/slope

# def amplitude(x,noise=-80,slope=35):
#    return 10**((x*slope+noise+slope)/20.)

def amplitude(x,noise_db=-60,max_db=35,trim_noise=False):
   if trim_noise:
      x_db = (x+1)/2*(max_db-noise_db)+noise_db
      if type(x) is np.ndarray:
         return 10**(x_db/10)*(np.sign(x_db-noise_db)*0.5+0.5)
      else:
         return 10**(x_db/10)*((x_db-noise_db).sign()*0.5+0.5)
   else:
      return 10**(((x+1)/2*(max_db-noise_db)+noise_db)/10)

def to_db(x,noise_db=-60,max_db=35):
   return (torchaudio.transforms.AmplitudeToDB()(x)-noise_db)/(max_db-noise_db)*2-1

def wave2spec(wave,n_fft=256,wave_fr=16000,spec_fr=125,noise_db=-60,max_db=22.5,to_db=True,power=2):
# def wave2spec(wave,n_fft=256,wave_fr=16000,spec_fr=125,noise_db=-50,max_db=22.5,to_db=True):
   if to_db:
      return (torchaudio.transforms.AmplitudeToDB()(torchaudio.transforms.Spectrogram(n_fft*2-1,win_length=n_fft*2-1,hop_length=int(wave_fr/spec_fr),power=power)(wave)).clamp(min=noise_db,max=max_db).transpose(-2,-1)-noise_db)/(max_db-noise_db)*2-1
   else:
      return torchaudio.transforms.Spectrogram(n_fft*2-1,win_length=n_fft*2-1,hop_length=int(wave_fr/spec_fr),power=power)(wave).transpose(-2,-1)


# def mel_scale(n_mels,hz,min_octave=-31.,max_octave=95.,pt=True):
# def mel_scale(n_mels,hz,min_octave=-58.,max_octave=100.,pt=True):
def mel_scale(n_mels,hz,min_octave=-31.,max_octave=102.,pt=True):
    #take absolute hz, return abs mel
   #  return (torch.log2(hz/440)+31/24)*24*n_mels/126
   if pt:
      return (torch.log2(hz/440.)-min_octave/24.)*24*n_mels/(max_octave-min_octave)
   else:
      return (np.log2(hz/440.)-min_octave/24.)*24*n_mels/(max_octave-min_octave)

# def inverse_mel_scale(mel,min_octave=-31.,max_octave=95.):
# def inverse_mel_scale(mel,min_octave=-58.,max_octave=100.):
def inverse_mel_scale(mel,min_octave=-31.,max_octave=102.):
    #take normalized mel, return absolute hz
   #  return 440*2**(mel*126/24-31/24)
   return 440*2**(mel*(max_octave-min_octave)/24.+min_octave/24.)

# def mel_scale(n_mels,hz,f_min=160.,f_max=8000.,pt=True):
#     #take absolute hz, return abs mel
#    #  return (torch.log2(hz/440)+31/24)*24*n_mels/126
#    m_min = 2595.0 * np.log10(1.0 + (f_min / 700.0))
#    m_max = 2595.0 * np.log10(1.0 + (f_max / 700.0))
#    m_min_ = m_min + (m_max-m_min)/(n_mels+1)
#    m_max_ = m_max
#    if pt:
#       return (2595.0 * torch.log10(1.0 + (hz / 700.0))-m_min_)/(m_max_-m_min_)*n_mels
#    else:
#       return (2595.0 * np.log10(1.0 + (hz / 700.0))-m_min_)/(m_max_-m_min_)*n_mels

# def inverse_mel_scale(mel,f_min=160.,f_max=8000.,n_mels=64):
#     #take normalized mel, return absolute hz
#    #  return 440*2**(mel*126/24-31/24)
#    m_min = 2595.0 * np.log10(1.0 + (f_min / 700.0))
#    m_max = 2595.0 * np.log10(1.0 + (f_max / 700.0))
#    m_min_ = m_min + (m_max-m_min)/(n_mels+1)
#    m_max_ = m_max
#    return 700.0 * (10**((mel*(m_max_-m_min_) + m_min_)/ 2595.0) - 1.0)

def ind2hz(ind,n_fft,max_freq=8000.):
   #input abs ind, output abs hz
   return ind/(1.0*n_fft)*max_freq

def hz2ind(hz,n_fft,max_freq=8000.):
   # input abs hz, output abs ind
   return hz/(1.0*max_freq)*n_fft

def bandwidth_mel(freqs_hz,bandwidth_hz,n_mels):
   # input hz bandwidth, output abs bandwidth on mel
   bandwidth_upper = freqs_hz+bandwidth_hz/2.
   bandwidth_lower = torch.clamp(freqs_hz-bandwidth_hz/2.,min=1)
   bandwidth = mel_scale(n_mels,bandwidth_upper) - mel_scale(n_mels,bandwidth_lower)
   return bandwidth

def torch_P2R(radii, angles):
   return radii * torch.cos(angles),radii * torch.sin(angles)
def inverse_spec_to_audio(spec,n_fft = 511,win_length = 511,hop_length = 128,power_synth=True):
   '''
   generate random phase, then use istft to inverse spec to audio
   '''
   window = torch.hann_window(win_length)
   angles = torch.randn_like(spec).uniform_(0, np.pi*2)#torch.zeros_like(spec)#torch.randn_like(spec).uniform_(0, np.pi*2)
   spec = spec**0.5 if power_synth else spec
   spec_complex = torch.stack(torch_P2R(spec, angles),dim=-1) #real and image in same dim
   return torchaudio.functional.istft(spec_complex, n_fft=n_fft, window=window, center=True, win_length=win_length, hop_length=hop_length)

@GENERATORS.register("GeneratorFormant")
class FormantSysth(nn.Module):
   def __init__(self, n_mels=64, k=100, wavebased=False,n_fft=256,noise_db=-50,max_db=22.5,dbbased=False,add_bgnoise=True,log10=False,noise_from_data=False,return_wave=False,power_synth=False):
      super(FormantSysth, self).__init__()
      self.wave_fr = 16e3
      self.spec_fr = 125
      self.n_fft = n_fft
      self.noise_db=noise_db
      self.max_db = max_db
      self.n_mels = n_mels
      self.k = k
      self.dbbased=dbbased
      self.log10 = log10
      self.add_bgnoise = add_bgnoise
      self.wavebased=wavebased
      self.noise_from_data = noise_from_data
      self.linear_scale = wavebased
      self.return_wave = return_wave
      self.power_synth = power_synth
      self.timbre = Parameter(torch.Tensor(1,1,n_mels))
      self.timbre_mapping = nn.Sequential(
                                 ln.Conv1d(1,128,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,128,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,2,1),
                                 # nn.Sigmoid(),
      )
      self.bgnoise_mapping = nn.Sequential(
                                 ln.Conv2d(2,2,[1,5],padding=[0,2],gain=1,bias=False),
                                 # nn.Sigmoid(),
      )
      self.noise_mapping = nn.Sequential(
                                 ln.Conv2d(2,2,[1,5],padding=[0,2],gain=1,bias=False),
                                 # nn.Sigmoid(),
      )

      self.bgnoise_mapping2 = nn.Sequential(
                                 ln.Conv1d(1,128,1,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,128,1,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,1,1,gain=1,bias=False),
                                 # nn.Sigmoid(),
      )
      self.noise_mapping2 = nn.Sequential(
                                 ln.Conv1d(1,128,1,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,128,1,1),
                                 nn.LeakyReLU(0.2),
                                 ln.Conv1d(128,1,1,1,gain=1,bias=False),
                                 # nn.Sigmoid(),
      )
      self.prior_exp = np.array([0.4963,0.0745,1.9018])
      self.timbre_parameter = Parameter(torch.Tensor(2))
      self.wave_noise_amplifier = Parameter(torch.Tensor(1))
      self.wave_hamon_amplifier = Parameter(torch.Tensor(1))

      if noise_from_data:
         self.bgnoise_amp = Parameter(torch.Tensor(1))
         with torch.no_grad():
            nn.init.constant_(self.bgnoise_amp,1)
      else:
         self.bgnoise_dist = Parameter(torch.Tensor(1,1,1,self.n_fft if self.wavebased else self.n_mels))
         with torch.no_grad():
            nn.init.constant_(self.bgnoise_dist,1.0)
    #   self.silient = Parameter(torch.Tensor(1,1,n_mels))
      self.silient = -1
      with torch.no_grad():
         nn.init.constant_(self.timbre,1.0)
         nn.init.constant_(self.timbre_parameter[0],7)
         nn.init.constant_(self.timbre_parameter[1],0.004)
         nn.init.constant_(self.wave_noise_amplifier,1)
         nn.init.constant_(self.wave_hamon_amplifier,4.)
         
        #  nn.init.constant_(self.silient,-1.0)

#    def formant_mask(self,freq,bandwith,amplitude):
#       # freq, bandwith, amplitude: B*formants*time
#       freq_cord = torch.arange(self.n_mels)
#       time_cord = torch.arange(freq.shape[2])
#       grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
#       grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
#       grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
#       freq = freq.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
#       bandwith = bandwith.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
#       amplitude = amplitude.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
#       # masks = amplitude*torch.exp(-0.693*(grid_freq-freq)**2/(2*(bandwith+0.001)**2)) #B,time,freqchans, formants
#       masks = amplitude*torch.exp(-(grid_freq-freq)**2/(2*(bandwith/np.sqrt(2*np.log(2))+0.001)**2)) #B,time,freqchans, formants
#       masks = masks.unsqueeze(dim=1) #B,1,time,freqchans, formants
#       return masks
   
   def formant_mask(self,freq_hz,bandwith_hz,amplitude,linear=False, triangle_mask = False,duomask=True, n_formant_noise=1,f0_hz=None,noise=False):
      # freq, bandwith, amplitude: B*formants*time
      freq_cord = torch.arange(self.n_fft if linear else self.n_mels)
      time_cord = torch.arange(freq_hz.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq_hz = ind2hz(grid_freq,self.n_fft,self.wave_fr/2) if linear else inverse_mel_scale(grid_freq/(self.n_mels*1.0))
      freq_hz = freq_hz.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      bandwith_hz = bandwith_hz.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      amplitude = amplitude.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      if self.power_synth:
         amplitude = amplitude
      alpha = (2*np.sqrt(2*np.log(np.sqrt(2))))
      if not noise:
         # t = torch.arange(int(f0_hz.shape[2]/self.spec_fr*self.wave_fr))/(1.0*self.wave_fr) #in second
         # t = t.unsqueeze(dim=0).unsqueeze(dim=0) #1, 1, time
         k = (torch.arange(self.k)+1).reshape([1,self.k,1])
         # f0_hz_interp = F.interpolate(f0_hz,t.shape[-1],mode='linear',align_corners=False) #Bx1xT
         # bandwith_hz_interp = F.interpolate(bandwith_hz.permute(0,2,3,1),[bandwith_hz.shape[-1],t.shape[-1]],mode='bilinear',align_corners=False).permute(0,3,1,2) #Bx1xT
         # freq_hz_interp = F.interpolate(freq_hz.permute(0,2,3,1),[freq_hz.shape[-1],t.shape[-1]],mode='bilinear',align_corners=False).permute(0,3,1,2) #Bx1xT
         k_f0 = k*f0_hz #BxkxT
         freq_range = (-torch.sign(k_f0-7800)*0.5+0.5) #BxkxT
         k_f0 = k_f0.permute([0,2,1]).unsqueeze(-1) #BxTxkx1
    #   masks = amplitude*torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2)) if self.wavebased else amplitude*torch.exp(-(0.693*(grid_freq_hz-freq_hz))**2/(2*(bandwith_hz+0.01)**2)) #B,time,freqchans, formants
         # amplitude_interp = F.interpolate(amplitude.permute(0,2,3,1),[amplitude.shape[-1],t.shape[-1]],mode='bilinear',align_corners=False).permute(0,3,1,2) #Bx1xT
         hamonic_dist = (amplitude*(torch.exp(-((k_f0-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))+1E-6).sqrt()).sum(-1).permute([0,2,1]) #BxkxT
         hamonic_dist = (hamonic_dist*freq_range)/((((hamonic_dist*freq_range)**2).sum(1,keepdim=True)+1E-10).sqrt()+1E-10) # sum_k(hamonic_dist**2) = 1
         hamonic_dist = F.interpolate(hamonic_dist,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr),mode = 'linear',align_corners=False)
         return hamonic_dist # B,k,T
      else:
         masks = amplitude*(torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))+1E-6).sqrt() #B,time,freqchans, formants
         masks = masks.sum(-1) #B,time,freqchans
         masks = masks/((((masks**2).sum(-1,keepdim=True)/self.n_fft)+1E-10).sqrt()+1E-10)
         masks = masks.unsqueeze(dim=1) #B,1,time,freqchans
         return masks #B,1,time,freqchans

      # if self.wavebased:
      # if triangle_mask:
      #    if duomask:
      #       # masks_hamon = amplitude[...,:-n_formant_noise]*torch.exp(-(0.693*(grid_freq_hz-freq_hz[...,:-n_formant_noise]))**2/(2*(bandwith_hz[...,:-n_formant_noise]+0.01)**2))
      #       masks_hamon = amplitude[...,:-n_formant_noise]*torch.exp(-((grid_freq_hz-freq_hz[...,:-n_formant_noise]))**2/(2*(bandwith_hz[...,:-n_formant_noise]/alpha+0.01)**2))
      #       bw = bandwith_hz[...,-n_formant_noise:]
      #       masks_noise = F.relu(amplitude[...,-n_formant_noise:] * (1 - (1-1/np.sqrt(2))*2/(bw+0.01)*(grid_freq_hz-freq_hz[...,-n_formant_noise:]).abs()))
      #       # masks_noise = amplitude[...,-n_formant_noise:] * (1 - 2/(bw+0.01)*(grid_freq_hz-freq_hz[...,-n_formant_noise:]).abs())*(-torch.sign(torch.abs(grid_freq_hz-freq_hz[...,-n_formant_noise:])/(bw+0.01)-0.5)*0.5+0.5)
      #       masks = torch.cat([masks_hamon,masks_noise],dim=-1)
      #    else:
      #       masks = F.relu(amplitude * (1 - (1-1/np.sqrt(2))*2/(bandwith_hz+0.01)*(grid_freq_hz-freq_hz).abs()))
      # else:
      #    # masks = amplitude*torch.exp(-(0.693*(grid_freq_hz-freq_hz))**2/(2*(bandwith_hz+0.01)**2))
      #    if self.power_synth:
      #       masks = amplitude*torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))
      #    else:
      #       # masks = amplitude*torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))
      #       masks = amplitude*(torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))+1E-6).sqrt()
      # else:
      #    if triangle_mask:
      #       if duomask:
      #          # masks_hamon = amplitude[...,:-n_formant_noise]*torch.exp(-(0.693*(grid_freq_hz-freq_hz[...,:-n_formant_noise]))**2/(2*(bandwith_hz[...,:-n_formant_noise]+0.01)**2))
      #          masks_hamon = amplitude[...,:-n_formant_noise]*torch.exp(-((grid_freq_hz-freq_hz[...,:-n_formant_noise]))**2/(2*(bandwith_hz[...,:-n_formant_noise]/alpha+0.01)**2))
      #          bw = bandwith_hz[...,-n_formant_noise:]
      #          masks_noise = F.relu(amplitude[...,-n_formant_noise:] * (1 - (1-1/np.sqrt(2))*2/(bw+0.01)*(grid_freq_hz-freq_hz[...,-n_formant_noise:]).abs()))
      #          # masks_noise = amplitude[...,-n_formant_noise:] * (1 - 2/(bw+0.01)*(grid_freq_hz-freq_hz[...,-n_formant_noise:]).abs())*(-torch.sign(torch.abs(grid_freq_hz-freq_hz[...,-n_formant_noise:])/(bw+0.01)-0.5)*0.5+0.5)
      #          masks = torch.cat([masks_hamon,masks_noise],dim=-1)
      #       else:
      #          masks = F.relu(amplitude * (1 - (1-1/np.sqrt(2))*2/(bandwith_hz+0.01)*(grid_freq_hz-freq_hz).abs()))
      #       # masks = amplitude * (1 - 2/(bandwith_hz+0.01)*(grid_freq_hz-freq_hz).abs())*(-torch.sign(torch.abs(grid_freq_hz-freq_hz)/(bandwith_hz+0.01)-0.5)*0.5+0.5)
      #    else:
      #       # masks = amplitude*torch.exp(-(0.693*(grid_freq_hz-freq_hz))**2/(2*(bandwith_hz+0.01)**2))
      #       masks = amplitude*torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/alpha+0.01)**2))
      # masks = amplitude*torch.exp(-((grid_freq_hz-freq_hz))**2/(2*(bandwith_hz/(2*np.sqrt(2*np.log(2)))+0.01)**2)) #B,time,freqchans, formants
      # masks = amplitude*torch.exp(-(0.693*(grid_freq_hz-freq_hz))**2/(2*(bandwith_hz+0.01)**2)) #B,time,freqchans, formants

   def voicing_wavebased(self,f0_hz):
      #f0: B*1*time, hz
      t = torch.arange(int(f0_hz.shape[2]/self.spec_fr*self.wave_fr))/(1.0*self.wave_fr) #in second
      t = t.unsqueeze(dim=0).unsqueeze(dim=0) #1, 1, time
      k = (torch.arange(self.k)+1).reshape([1,self.k,1])
      f0_hz_interp = F.interpolate(f0_hz,t.shape[-1],mode='linear',align_corners=False)
      k_f0 = k*f0_hz_interp
      k_f0_sum = 2*np.pi*torch.cumsum(k_f0,-1)/(1.0*self.wave_fr)
      wave_k = np.sqrt(2)*torch.sin(k_f0_sum) * (-torch.sign(k_f0-7800)*0.5+0.5)
      # wave = 0.12*torch.sin(2*np.pi*k_f0*t) * (-torch.sign(k_f0-6000)*0.5+0.5)
      # wave = 0.09*torch.sin(2*np.pi*k_f0*t) * (-torch.sign(k_f0-self.wave_fr/2)*0.5+0.5)
      # wave = 0.09*torch.sigmoid(self.wave_hamon_amplifier) * torch.sin(2*np.pi*k_f0*t) * (-torch.sign(k_f0-self.wave_fr/2)*0.5+0.5)
      # wave = wave_k.sum(dim=1,keepdim=True)
      # wave = F.softplus(self.wave_hamon_amplifier) * wave.sum(dim=1,keepdim=True)
      # spec = wave2spec(wave,self.n_fft,self.wave_fr,self.spec_fr,self.noise_db,self.max_db,to_db=self.dbbased,power=2. if self.power_synth else 1.)
      return wave_k #B,k,T
      # if self.return_wave:
      #    return spec,wave_k
      # else:
      #    return spec
   
   def unvoicing_wavebased(self,f0_hz,bg=False,mapping=True):
      # return torch.ones([1,1,f0_hz.shape[2],512])
      # noise = 0.3*torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
      # noise = 0.03*torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
      if bg:
         noise = torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
         if mapping:
            noise = self.bgnoise_mapping2(noise)
      else:
         noise = np.sqrt(3.)*(2*torch.rand([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])-1)
         if mapping:
            noise = self.noise_mapping2(noise)
      # noise = 0.3 * torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
    #   noise = 0.3 * F.softplus(self.wave_noise_amplifier) * torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
      return wave2spec(noise,self.n_fft,self.wave_fr,self.spec_fr,self.noise_db,self.max_db,to_db=False,power=2. if self.power_synth else 1.)
      # return torchaudio.transforms.Spectrogram(self.n_fft*2-1,win_length=self.n_fft*2-1,hop_length=int(self.wave_fr/self.spec_fr),power=2. if self.power_synth else 1.)(noise)
   
   # def unvoicing_wavebased(self,f0_hz):
   #    # return torch.ones([1,1,f0_hz.shape[2],512])
   #    # noise = 0.3*torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
   #    noise = 0.1*torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
   #    # noise = 0.3 * torch.sigmoid(self.wave_noise_amplifier) * torch.randn([1,1,int(f0_hz.shape[2]/self.spec_fr*self.wave_fr)])
   #    return wave2spec(noise,self.n_fft,self.wave_fr,self.spec_fr,self.noise_db,self.max_db,to_db=False)
      
   def voicing_linear(self,f0_hz,bandwith=2.5):
      #f0: B*1*time, hz
      freq_cord = torch.arange(self.n_fft)
      time_cord = torch.arange(f0_hz.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      f0_hz = f0_hz.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, 1
      f0_hz = f0_hz.repeat([1,1,1,self.k]) #B,time,1, self.k
      f0_hz = f0_hz*(torch.arange(self.k)+1).reshape([1,1,1,self.k])
      # bandwith=4
    #   bandwith_lower = torch.clamp(f0-bandwith/2,min=1)
    #   bandwith_upper = f0+bandwith/2
    #   bandwith = mel_scale(self.n_mels,bandwith_upper) - mel_scale(self.n_mels,bandwith_lower)
      f0 = hz2ind(f0_hz,self.n_fft)

      # hamonics = torch.exp(-(grid_freq-f0)**2/(2*sigma**2)) #gaussian
      # hamonics = (1-((grid_freq_hz-f0_hz)/(2*bandwith_hz/2))**2)*(-torch.sign(torch.abs(grid_freq_hz-f0_hz)/(2*bandwith_hz)-0.5)*0.5+0.5) #welch
      freq_cord_reshape = freq_cord.reshape([1,1,1,self.n_fft])
      hamonics = (1 - 2/bandwith*(grid_freq-f0).abs())*(-torch.sign(torch.abs(grid_freq-f0)/(bandwith)-0.5)*0.5+0.5) #triangular
      # hamonics = (1-((grid_freq-f0)/(bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(bandwith)-0.5)*0.5+0.5) #welch
      # hamonics = (1-((grid_freq-f0)/(2.5*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(2.5*bandwith)-0.5)*0.5+0.5) #welch
      # hamonics = (1-((grid_freq-f0)/(3*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(3*bandwith)-0.5)*0.5+0.5) #welch
      # hamonics = torch.cos(np.pi*torch.abs(grid_freq-f0)/(4*bandwith))**2*(-torch.sign(torch.abs(grid_freq-f0)/(4*bandwith)-0.5)*0.5+0.5) #hanning
      # hamonics = (hamonics.sum(dim=-1)).unsqueeze(dim=1) # B,1,T,F
      # condition = (torch.sign(freq_cord_reshape-switch)*0.5+0.5)
      # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-slop*(freq_cord_reshape-switch)*condition)-1)*condition) # B,1,T,F
      # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)-torch.abs(self.prior_exp_parameter[2])) * torch.exp(-torch.abs(self.prior_exp_parameter[1])*freq_cord.reshape([1,1,1,self.n_mels])) + torch.abs(self.prior_exp_parameter[2]) # B,1,T,F
      
      # timbre_parameter = self.timbre_mapping(f0_hz[...,0,0].unsqueeze(1)).permute([0,2,1]).unsqueeze(1)
      # condition = (torch.sign(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*0.5+0.5)
      # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-0.01*torch.sigmoid(timbre_parameter[...,1:2])*(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*condition)-1)*condition) # B,1,T,F
      # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-0.01*torch.sigmoid(timbre_parameter[...,1:2])*(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*condition)-1)*condition) * F.softplus(timbre_parameter[...,2:3]) + timbre_parameter[...,3:4] # B,1,T,F
      # timbre = self.timbre_mapping(f0_hz[...,0,0].unsqueeze(1)).permute([0,2,1])
      # hamonics = (hamonics.sum(dim=-1)*timbre).unsqueeze(dim=1) # B,1,T,F
      # hamonics = (hamonics.sum(dim=-1)*self.timbre).unsqueeze(dim=1) # B,1,T,F

      hamonics = (hamonics.sum(dim=-1)).unsqueeze(dim=1)
      # hamonics = 180*F.softplus(self.wave_hamon_amplifier)*(hamonics.sum(dim=-1)).unsqueeze(dim=1)

      return hamonics

   def voicing(self,f0_hz):
      #f0: B*1*time, hz
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(f0_hz.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq_hz = inverse_mel_scale(grid_freq/(self.n_mels*1.0))
      f0_hz = f0_hz.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, 1
      f0_hz = f0_hz.repeat([1,1,1,self.k]) #B,time,1, self.k
      f0_hz = f0_hz*(torch.arange(self.k)+1).reshape([1,1,1,self.k])
      if self.log10:
         f0_mel = mel_scale(self.n_mels,f0_hz)
         band_low_hz = inverse_mel_scale((f0_mel-1)/(self.n_mels*1.0),n_mels = self.n_mels)
         band_up_hz = inverse_mel_scale((f0_mel+1)/(self.n_mels*1.0),n_mels = self.n_mels)
         bandwith_hz = band_up_hz-band_low_hz
         band_low_mel = mel_scale(self.n_mels,band_low_hz)
         band_up_mel = mel_scale(self.n_mels,band_up_hz)
         bandwith = band_up_mel-band_low_mel
      else:
         bandwith_hz = 24.7*(f0_hz*4.37/1000+1)
         bandwith = bandwidth_mel(f0_hz,bandwith_hz,self.n_mels)
    #   bandwith_lower = torch.clamp(f0-bandwith/2,min=1)
    #   bandwith_upper = f0+bandwith/2
    #   bandwith = mel_scale(self.n_mels,bandwith_upper) - mel_scale(self.n_mels,bandwith_lower)
      f0 = mel_scale(self.n_mels,f0_hz) 
      switch = mel_scale(self.n_mels,torch.abs(self.timbre_parameter[0])*f0_hz[...,0]).unsqueeze(1)
      slop = (torch.abs(self.timbre_parameter[1])*f0_hz[...,0]).unsqueeze(1)
      freq_cord_reshape = freq_cord.reshape([1,1,1,self.n_mels])
      if not self.dbbased:
         # sigma = bandwith/(np.sqrt(2*np.log(2)));  
         sigma = bandwith/(2*np.sqrt(2*np.log(2)));  
         hamonics = torch.exp(-(grid_freq-f0)**2/(2*sigma**2)) #gaussian
      # hamonics = (1-((grid_freq_hz-f0_hz)/(2*bandwith_hz/2))**2)*(-torch.sign(torch.abs(grid_freq_hz-f0_hz)/(2*bandwith_hz)-0.5)*0.5+0.5) #welch
      else:
      # #   hamonics = (1-((grid_freq-f0)/(1.75*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(1.75*bandwith)-0.5)*0.5+0.5) #welch
         hamonics = (1-((grid_freq-f0)/(2.5*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(2.5*bandwith)-0.5)*0.5+0.5) #welch
         # hamonics = (1-((grid_freq-f0)/(3*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(3*bandwith)-0.5)*0.5+0.5) #welch
         # hamonics = torch.cos(np.pi*torch.abs(grid_freq-f0)/(4*bandwith))**2*(-torch.sign(torch.abs(grid_freq-f0)/(4*bandwith)-0.5)*0.5+0.5) #hanning
         # hamonics = (hamonics.sum(dim=-1)).unsqueeze(dim=1) # B,1,T,F
         # condition = (torch.sign(freq_cord_reshape-switch)*0.5+0.5)
         # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-slop*(freq_cord_reshape-switch)*condition)-1)*condition) # B,1,T,F
         # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)-torch.abs(self.prior_exp_parameter[2])) * torch.exp(-torch.abs(self.prior_exp_parameter[1])*freq_cord.reshape([1,1,1,self.n_mels])) + torch.abs(self.prior_exp_parameter[2]) # B,1,T,F
      
      timbre_parameter = self.timbre_mapping(f0_hz[...,0,0].unsqueeze(1)).permute([0,2,1]).unsqueeze(1)
      condition = (torch.sign(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*0.5+0.5)
      amp = F.softplus(self.wave_hamon_amplifier) if self.dbbased else 180*F.softplus(self.wave_hamon_amplifier)
      hamonics = amp * ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-0.01*torch.sigmoid(timbre_parameter[...,1:2])*(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*condition)-1)*condition) # B,1,T,F
      # hamonics = ((hamonics.sum(dim=-1)).unsqueeze(dim=1)) * (1+ (torch.exp(-0.01*torch.sigmoid(timbre_parameter[...,1:2])*(freq_cord_reshape-torch.sigmoid(timbre_parameter[...,0:1])*self.n_mels)*condition)-1)*condition) * F.softplus(timbre_parameter[...,2:3]) + timbre_parameter[...,3:4] # B,1,T,F
      # timbre = self.timbre_mapping(f0_hz[...,0,0].unsqueeze(1)).permute([0,2,1])
      # hamonics = (hamonics.sum(dim=-1)*timbre).unsqueeze(dim=1) # B,1,T,F
      # hamonics = (hamonics.sum(dim=-1)*self.timbre).unsqueeze(dim=1) # B,1,T,F
      # return F.softplus(self.wave_hamon_amplifier)*hamonics
      return hamonics

   def unvoicing(self,f0,bg=False,mapping=True): 
    #   return (0.25*torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels]))+1
      rnd = torch.randn([f0.shape[0],2,f0.shape[2],self.n_fft if self.wavebased else self.n_mels])
      if mapping:
         rnd = self.bgnoise_mapping(rnd) if bg else self.noise_mapping(rnd)
      real = rnd[:,0:1]
      img = rnd[:,1:2]
      if self.dbbased:
         return (2*torchaudio.transforms.AmplitudeToDB()(torch.sqrt(real**2 + img**2+1E-10))+80).clamp(min=0)/35
         # return (2*torchaudio.transforms.AmplitudeToDB()(F.softplus(self.wave_noise_amplifier)*torch.sqrt(torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels])**2 + torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels])**2))+80).clamp(min=0)/35
      else:
         # return torch.ones([f0.shape[0],1,f0.shape[2],self.n_mels])
         return 180*F.softplus(self.wave_noise_amplifier) * torch.sqrt(real**2 + img**2+1E-10)
         # return F.softplus(self.wave_noise_amplifier)*torch.sqrt(torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels])**2 + torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels])**2)

      # return (F.softplus(self.wave_noise_amplifier)) * (0.25*torch.randn([f0.shape[0],1,f0.shape[2],self.n_mels]))+1
    #   return torch.ones([f0.shape[0],1,f0.shape[2],self.n_mels])

   def forward(self,components,enable_hamon_excitation=True,enable_noise_excitation=True,enable_bgnoise=True):
      # f0: B*1*T, amplitudes: B*2(voicing,unvoicing)*T, freq_formants,bandwidth_formants,amplitude_formants: B*formants*T
      amplitudes = components['amplitudes'].unsqueeze(dim=-1)
      amplitudes_h = components['amplitudes_h'].unsqueeze(dim=-1)
      loudness = components['loudness'].unsqueeze(dim=-1)
      f0_hz = components['f0_hz']
      # import pdb;pdb.set_trace()
      if self.wavebased:
         # self.hamonics = 1800*F.softplus(self.wave_hamon_amplifier)*self.voicing_linear(f0_hz)
         # self.noise = 180*self.unvoicing(f0_hz,bg=False,mapping=False)
         # self.bgnoise = 18*self.unvoicing(f0_hz,bg=True,mapping=False)
         # import pdb;pdb.set_trace()
         self.hamonics_wave = self.voicing_wavebased(f0_hz)
         self.noise = self.unvoicing_wavebased(f0_hz,bg=False,mapping=False)
         self.bgnoise = self.unvoicing_wavebased(f0_hz,bg=True)
      else: 
         self.hamonics = self.voicing(f0_hz)
         self.noise = self.unvoicing(f0_hz,bg=False)
         self.bgnoise = self.unvoicing(f0_hz,bg=True)
    #   freq_formants = components['freq_formants']*self.n_mels
    #   bandwidth_formants = components['bandwidth_formants']*self.n_mels
      # excitation = amplitudes[:,0:1]*hamonics
      # excitation = loudness*(amplitudes[:,0:1]*hamonics)
      
      self.excitation_noise = loudness*(amplitudes[:,-1:])*self.noise if self.power_synth else (loudness*amplitudes[:,-1:]+1E-10).sqrt()*self.noise
      duomask = components['freq_formants_noise_hz'].shape[1]>components['freq_formants_hamon_hz'].shape[1]
      n_formant_noise = (components['freq_formants_noise_hz'].shape[1]-components['freq_formants_hamon_hz'].shape[1]) if duomask else components['freq_formants_noise_hz'].shape[1]
      self.hamonic_dist = self.formant_mask(components['freq_formants_hamon_hz'],components['bandwidth_formants_hamon_hz'],components['amplitude_formants_hamon'],linear = self.linear_scale,f0_hz = f0_hz)
      self.mask_noise = self.formant_mask(components['freq_formants_noise_hz'],components['bandwidth_formants_noise_hz'],components['amplitude_formants_noise'],linear = self.linear_scale,triangle_mask=False if self.wavebased else True,duomask=duomask,n_formant_noise=n_formant_noise,f0_hz = f0_hz,noise=True)
      # self.mask_hamon = self.formant_mask(components['freq_formants_hamon']*self.n_mels,components['bandwidth_formants_hamon'],components['amplitude_formants_hamon'])
      # self.mask_noise = self.formant_mask(components['freq_formants_noise']*self.n_mels,components['bandwidth_formants_noise'],components['amplitude_formants_noise'])
      if self.power_synth:
         self.excitation_hamon_wave = F.interpolate(loudness[...,-1]*amplitudes[:,0:1][...,-1],self.hamonics_wave.shape[-1],mode='linear',align_corners=False)*self.hamonics_wave
      else:
         self.excitation_hamon_wave = F.interpolate((loudness[...,-1]*amplitudes[:,0:1][...,-1]+1E-10).sqrt(),self.hamonics_wave.shape[-1],mode='linear',align_corners=False)*self.hamonics_wave
      self.hamonics_wave_ = (self.excitation_hamon_wave*self.hamonic_dist).sum(1,keepdim=True)
 
      bgdist = F.softplus(self.bgnoise_amp)*self.noise_dist if self.noise_from_data else F.softplus(self.bgnoise_dist)
      # if self.power_synth:
      #    self.excitation_hamon = loudness*(amplitudes[:,0:1])*self.hamonics
      # else:
      #    self.excitation_hamon = loudness*amplitudes[:,0:1]*self.hamonics
      # import pdb;pdb.set_trace()
      self.noise_excitation = self.excitation_noise*self.mask_noise

      self.noise_excitation_wave = 2*inverse_spec_to_audio(self.noise_excitation.squeeze(1).permute(0,2,1),n_fft=self.n_fft*2-1,power_synth=self.power_synth)
      self.noise_excitation_wave = F.pad(self.noise_excitation_wave,[0,self.hamonics_wave_.shape[2]-self.noise_excitation_wave.shape[1]])
      self.noise_excitation_wave = self.noise_excitation_wave.unsqueeze(1)
      self.rec_wave_clean = self.noise_excitation_wave+self.hamonics_wave_

      if self.add_bgnoise and enable_bgnoise:
         self.bgn = bgdist*self.bgnoise*0.0003
         self.bgn_wave = 2*inverse_spec_to_audio(self.bgn.squeeze(1).permute(0,2,1),n_fft=self.n_fft*2-1,power_synth=self.power_synth)
         self.bgn_wave = F.pad(self.bgn_wave,[0,self.hamonics_wave_.shape[2]-self.bgn_wave.shape[1]])
         self.bgn_wave = self.bgn_wave.unsqueeze(1)
         self.rec_wave = self.rec_wave_clean + self.bgn_wave
      else:
         self.rec_wave = self.rec_wave_clean

      speech = wave2spec(self.rec_wave,self.n_fft,self.wave_fr,self.spec_fr,self.noise_db,self.max_db,to_db=True,power=2 if self.power_synth else 1)
      # if self.wavebased:
      #    # import pdb; pdb.set_trace()
      #    bgn = bgdist*self.bgnoise*0.0003 if (self.add_bgnoise and enable_bgnoise) else 0
      #    speech = ((self.excitation_hamon*self.mask_hamon_sum) if enable_hamon_excitation else torch.zeros(self.excitation_hamon.shape)) + (self.noise_excitation if enable_noise_excitation else 0) + bgn
      #    # speech = ((self.excitation_hamon*self.mask_hamon_sum) if enable_hamon_excitation else torch.zeros(self.excitation_hamon.shape)) + (self.excitation_noise*self.mask_noise_sum if enable_noise_excitation else 0) + (((bgdist*self.bgnoise*0.0003) if not self.dbbased else (2*torchaudio.transforms.AmplitudeToDB()(bgdist*0.0003)/35. + self.bgnoise)) if (self.add_bgnoise and enable_bgnoise) else 0)
      #    # speech = speech if self.power_synth else speech**2
      #    speech = (torchaudio.transforms.AmplitudeToDB()(speech).clamp(min=self.noise_db)-self.noise_db)/(self.max_db-self.noise_db)*2-1
      # else:
      #    # speech = ((self.excitation_hamon*self.mask_hamon_sum) if enable_hamon_excitation else torch.zeros(self.excitation_hamon.shape)) + (((bgdist*self.bgnoise*0.0003) if not self.dbbased else (2*torchaudio.transforms.AmplitudeToDB()(bgdist*0.0003)/35. + self.bgnoise)) if (self.add_bgnoise and enable_bgnoise) else 0) + (self.silient*torch.ones(self.mask_hamon_sum.shape) if self.dbbased else 0)
      #    speech = ((self.excitation_hamon*self.mask_hamon_sum) if enable_hamon_excitation else torch.zeros(self.excitation_hamon.shape)) + (self.noise_excitation if enable_noise_excitation else 0) + (((bgdist*self.bgnoise*0.0003) if not self.dbbased else (2*torchaudio.transforms.AmplitudeToDB()(bgdist*0.0003)/35. + self.bgnoise)) if (self.add_bgnoise and enable_bgnoise) else 0) + (self.silient*torch.ones(self.mask_hamon_sum.shape) if self.dbbased else 0)
      #    #  speech = self.excitation_hamon*self.mask_hamon_sum + (self.excitation_noise*self.mask_noise_sum if enable_noise_excitation else 0) + self.silient*torch.ones(self.mask_hamon_sum.shape)
      #    if not self.dbbased:
      #       speech = db(speech)

        
      # import pdb;pdb.set_trace() 
      if self.return_wave:
         return speech,self.rec_wave_clean
      else:
         return speech

@ENCODERS.register("EncoderFormant")
class FormantEncoder(nn.Module):
   def __init__(self, n_mels=64, n_formants=4,n_formants_noise=2,min_octave=-31,max_octave=96,wavebased=False,hop_length=128,n_fft=256,noise_db=-50,max_db=22.5,broud=True,power_synth=False):
      super(FormantEncoder, self).__init__()
      self.wavebased = wavebased
      self.n_mels = n_mels
      self.n_formants = n_formants
      self.n_formants_noise = n_formants_noise
      self.min_octave = min_octave
      self.max_octave = max_octave
      self.noise_db = noise_db
      self.max_db = max_db
      self.broud = broud
      self.hop_length = hop_length
      self.n_fft = n_fft
      self.power_synth=power_synth
      self.formant_freq_limits_diff = torch.tensor([950.,2450.,2100.]).reshape([1,3,1]) #freq difference
      self.formant_freq_limits_diff_low = torch.tensor([300.,300.,0.]).reshape([1,3,1]) #freq difference

      # self.formant_freq_limits_abs = torch.tensor([950.,2800.,3400.,4700.]).reshape([1,4,1]) #freq difference
      # self.formant_freq_limits_abs_low = torch.tensor([300.,600.,2800.,3400]).reshape([1,4,1]) #freq difference

      # self.formant_freq_limits_abs = torch.tensor([950.,3300.,3600.,4700.]).reshape([1,4,1]) #freq difference
      # self.formant_freq_limits_abs_low = torch.tensor([300.,600.,2700.,3400]).reshape([1,4,1]) #freq difference

      self.formant_freq_limits_abs = torch.tensor([950.,3400.,3800.,5000.,6000.,7000.]).reshape([1,6,1]) #freq difference
      self.formant_freq_limits_abs_low = torch.tensor([300.,700.,1800.,3400,5000.,6000.]).reshape([1,6,1]) #freq difference
      # self.formant_freq_limits_abs_low = torch.tensor([300.,700.,2700.,3400]).reshape([1,4,1]) #freq difference

      # self.formant_freq_limits_abs_noise = torch.tensor([7000.]).reshape([1,1,1]) #freq difference
      self.formant_freq_limits_abs_noise = torch.tensor([8000.,7000.,7000.]).reshape([1,3,1]) #freq difference
      self.formant_freq_limits_abs_noise_low = torch.tensor([4000.,3000.,3000.]).reshape([1,3,1]) #freq difference
      # self.formant_freq_limits_abs_noise_low = torch.tensor([4000.,500.,500.]).reshape([1,3,1]) #freq difference

      self.formant_bandwitdh_bias = Parameter(torch.Tensor(1))
      self.formant_bandwitdh_slop = Parameter(torch.Tensor(1))
      self.formant_bandwitdh_thres = Parameter(torch.Tensor(1))
      with torch.no_grad():
         nn.init.constant_(self.formant_bandwitdh_bias,0)
         nn.init.constant_(self.formant_bandwitdh_slop,0)
         nn.init.constant_(self.formant_bandwitdh_thres,0)

    #   self.formant_freq_limits = torch.cumsum(self.formant_freq_limits_diff,dim=0)
    #   self.formant_freq_limits_mel = torch.cat([torch.tensor([0.]),mel_scale(n_mels,self.formant_freq_limits)/n_mels])
    #   self.formant_freq_limits_mel_diff = torch.reshape(self.formant_freq_limits_mel[1:]-self.formant_freq_limits_mel[:-1],[1,3,1])
      if broud:
         if wavebased:
            self.conv1_narrow = ln.Conv1d(n_fft,64,3,1,1)
            self.conv1_mel = ln.Conv1d(128,64,3,1,1)
            self.norm1_mel = nn.GroupNorm(32,64)
            self.conv2_mel = ln.Conv1d(64,128,3,1,1)
            self.norm2_mel = nn.GroupNorm(32,128) 
            self.conv_fundementals_mel = ln.Conv1d(128,128,3,1,1)
            self.norm_fundementals_mel = nn.GroupNorm(32,128)
            self.f0_drop_mel = nn.Dropout()
         else:
            self.conv1_narrow = ln.Conv1d(n_mels,64,3,1,1)
         self.norm1_narrow = nn.GroupNorm(32,64)
         self.conv2_narrow = ln.Conv1d(64,128,3,1,1)
         self.norm2_narrow = nn.GroupNorm(32,128) 

         self.conv_fundementals_narrow = ln.Conv1d(128,128,3,1,1)
         self.norm_fundementals_narrow = nn.GroupNorm(32,128)
         self.f0_drop_narrow = nn.Dropout()
         if wavebased:
            self.conv_f0_narrow = ln.Conv1d(256,1,1,1,0)
         else:
            self.conv_f0_narrow = ln.Conv1d(128,1,1,1,0)

         self.conv_amplitudes_narrow = ln.Conv1d(128,2,1,1,0)
         self.conv_amplitudes_h_narrow = ln.Conv1d(128,2,1,1,0)
         
      if wavebased:
         self.conv1 = ln.Conv1d(n_fft,64,3,1,1)
      else:
         self.conv1 = ln.Conv1d(n_mels,64,3,1,1)
      self.norm1 = nn.GroupNorm(32,64)
      self.conv2 = ln.Conv1d(64,128,3,1,1)
      self.norm2 = nn.GroupNorm(32,128) 

      self.conv_fundementals = ln.Conv1d(128,128,3,1,1)
      self.norm_fundementals = nn.GroupNorm(32,128)
      self.f0_drop = nn.Dropout()
      self.conv_f0 = ln.Conv1d(128,1,1,1,0)

      self.conv_amplitudes = ln.Conv1d(128,2,1,1,0)
      self.conv_amplitudes_h = ln.Conv1d(128,2,1,1,0)
      # self.conv_loudness = nn.Sequential(ln.Conv1d(n_fft if wavebased else n_mels,128,1,1,0),
      #                                  nn.LeakyReLU(0.2),
      #                                  ln.Conv1d(128,128,1,1,0),
      #                                  nn.LeakyReLU(0.2),
      #                                  ln.Conv1d(128,1,1,1,0,bias_initial=0.5),)
      self.conv_loudness = nn.Sequential(ln.Conv1d(n_fft if wavebased else n_mels,128,1,1,0),
                                       nn.LeakyReLU(0.2),
                                       ln.Conv1d(128,128,1,1,0),
                                       nn.LeakyReLU(0.2),
                                       ln.Conv1d(128,1,1,1,0,bias_initial=-9.),)
      # self.conv_loudness_power = nn.Sequential(ln.Conv1d(1,128,1,1,0),
      #                                  nn.LeakyReLU(0.2),
      #                                  ln.Conv1d(128,128,1,1,0),
      #                                  nn.LeakyReLU(0.2),
      #                                  ln.Conv1d(128,1,1,1,0,bias_initial=-9.),)

      if self.broud:
         self.conv_formants = ln.Conv1d(128,128,3,1,1)
      else:
         self.conv_formants = ln.Conv1d(128,128,3,1,1)
      self.norm_formants = nn.GroupNorm(32,128)
      self.conv_formants_freqs = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_bandwidth = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_amplitude = ln.Conv1d(128,n_formants,1,1,0)

      self.conv_formants_freqs_noise = ln.Conv1d(128,self.n_formants_noise,1,1,0)
      self.conv_formants_bandwidth_noise = ln.Conv1d(128,self.n_formants_noise,1,1,0)
      self.conv_formants_amplitude_noise = ln.Conv1d(128,self.n_formants_noise,1,1,0)

      self.amplifier = Parameter(torch.Tensor(1))
      self.bias = Parameter(torch.Tensor(1))
      with torch.no_grad():
         nn.init.constant_(self.amplifier,1.0)
         nn.init.constant_(self.bias,0.)
   
   def forward(self,x,x_denoise=None,duomask=False,noise_level = None,x_amp=None):
      x = x.squeeze(dim=1).permute(0,2,1) #B * f * T
      if x_denoise is not None:
         x_denoise = x_denoise.squeeze(dim=1).permute(0,2,1)
         # x_denoise_amp = amplitude(x_denoise,self.noise_db,self.max_db)
      # import pdb; pdb.set_trace()
      if x_amp is None:
         x_amp = amplitude(x,self.noise_db,self.max_db,trim_noise=True)
      else:
         x_amp = x_amp.squeeze(dim=1).permute(0,2,1)
      hann_win = torch.hann_window(5,periodic=False).reshape([1,1,5,1])
      x_smooth = F.conv2d(x.unsqueeze(1).transpose(-2,-1),hann_win,padding=[2,0]).transpose(-2,-1).squeeze(1)
      # loudness = F.softplus(self.amplifier)*(torch.mean(x_denoise_amp,dim=1,keepdim=True))
      # loudness = F.relu(F.softplus(self.amplifier)*(torch.mean(x_amp,dim=1,keepdim=True)-noise_level*0.0003))
      # loudness = torch.mean((x*0.5+0.5) if x_denoise is None else (x_denoise*0.5+0.5),dim=1,keepdim=True)
      # loudness = F.softplus(self.amplifier)*(loudness)
      # loudness = F.softplus(self.amplifier)*torch.mean(x_amp,dim=1,keepdim=True)
      # loudness = F.softplus(self.amplifier)*F.relu(loudness - F.softplus(self.bias))
      if self.power_synth:
         loudness = F.softplus((1. if self.wavebased else 1.0)*self.conv_loudness(x_smooth))
      else:
         # loudness = F.softplus(self.amplifier)*F.relu((x_amp**2).sum(1,keepdim=True)/self.hop_length/383.-self.bias)
         loudness = F.softplus((1. if self.wavebased else 1.0)*self.conv_loudness(x_smooth)) # compute power loudness
         # loudness = F.softplus(self.amplifier)*F.relu((x_amp**2).sum(1,keepdim=True)-self.bias) # compute power loudness
      # loudness = F.softplus((1. if self.wavebased else 1.0)*self.conv_loudness(x))
      # loudness  = F.relu(self.conv_loudness(x))

      # if not self.power_synth:
      #    loudness = loudness.sqrt()

      if self.broud:
         x_narrow = x
         x_narrow = F.leaky_relu(self.norm1_narrow(self.conv1_narrow(x_narrow)),0.2)
         x_common_narrow = F.leaky_relu(self.norm2_narrow(self.conv2_narrow(x_narrow)),0.2)
         amplitudes = F.softmax(self.conv_amplitudes_narrow(x_common_narrow),dim=1)
         amplitudes_h = F.softmax(self.conv_amplitudes_h_narrow(x_common_narrow),dim=1)
         x_fundementals_narrow = self.f0_drop_narrow(F.leaky_relu(self.norm_fundementals_narrow(self.conv_fundementals_narrow(x_common_narrow)),0.2))
         
         x_amp = amplitude(x.unsqueeze(1),self.noise_db,self.max_db).transpose(-2,-1)
         x_mel = to_db(torchaudio.transforms.MelScale(f_max=8000,n_stft=self.n_fft)(x_amp.transpose(-2,-1)),self.noise_db,self.max_db).squeeze(1)
         x = F.leaky_relu(self.norm1_mel(self.conv1_mel(x_mel)),0.2)
         x_common_mel = F.leaky_relu(self.norm2_mel(self.conv2_mel(x)),0.2)
         x_fundementals_mel = self.f0_drop_mel(F.leaky_relu(self.norm_fundementals_mel(self.conv_fundementals_mel(x_common_mel)),0.2))

         f0_hz = torch.sigmoid(self.conv_f0_narrow(torch.cat([x_fundementals_narrow,x_fundementals_mel],dim=1))) * 120 + 180 # 180hz < f0 < 300 hz
         f0 = torch.clamp(mel_scale(self.n_mels,f0_hz)/(self.n_mels*1.0),min=0.0001)
         
         hann_win = torch.hann_window(21,periodic=False).reshape([1,1,21,1])
         x = to_db(F.conv2d(x_amp,hann_win,padding=[10,0]).transpose(-2,-1),self.noise_db,self.max_db).squeeze(1)
         x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
         x_common = F.leaky_relu(self.norm2(self.conv2(x)),0.2)

         
      
      else:
         x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
         x_common = F.leaky_relu(self.norm2(self.conv2(x)),0.2)

         
         # loudness  = F.relu(self.conv_loudness(x_common))
         # loudness = F.relu(self.conv_loudness(x_common)) +(10**(self.noise_db/10.-1) if self.wavebased else 0)
         amplitudes = F.softmax(self.conv_amplitudes(x_common),dim=1)
         amplitudes_h = F.softmax(self.conv_amplitudes_h(x_common),dim=1)

         # x_fundementals = F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2)
         x_fundementals = self.f0_drop(F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2))
         # f0 in mel:
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals))
         # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
      #   f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice

         # f0 in hz:
         # f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 120 + 180 # 180hz < f0 < 300 hz
         f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 332 + 88 # 88hz < f0 < 420 hz
      #   f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 120 + 180 # 180hz < f0 < 300 hz
         # f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 528 + 88 # 88hz < f0 < 616 hz
         # f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 302 + 118 # 118hz < f0 < 420 hz
      #   f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 240 + 180 # 180hz < f0 < 420 hz
         # f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 260 + 160 # 160hz < f0 < 420 hz
         f0 = torch.clamp(mel_scale(self.n_mels,f0_hz)/(self.n_mels*1.0),min=0.0001)

      x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
      formants_freqs = torch.sigmoid(self.conv_formants_freqs(x_formants))
    # #  relative freq:
    #   formants_freqs_hz = formants_freqs*(self.formant_freq_limits_diff[:,:self.n_formants]-self.formant_freq_limits_diff_low[:,:self.n_formants])+self.formant_freq_limits_diff_low[:,:self.n_formants]
    # #   formants_freqs_hz = formants_freqs*6839
    #   formants_freqs_hz = torch.cumsum(formants_freqs_hz,dim=1)

      # abs freq:
      formants_freqs_hz = formants_freqs*(self.formant_freq_limits_abs[:,:self.n_formants]-self.formant_freq_limits_abs_low[:,:self.n_formants])+self.formant_freq_limits_abs_low[:,:self.n_formants]
    #   formants_freqs_hz = formants_freqs*6839
      formants_freqs = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz)/(self.n_mels*1.0),min=0) 
      
      # formants_freqs = formants_freqs + f0
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) *6839
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 150
      # formants_bandwidth_hz = 3*torch.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.relu(formants_freqs_hz-1000)+100)
    #   formants_bandwidth_hz = (torch.sigmoid(self.conv_formants_bandwidth(x_formants))) * (3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.075*torch.relu(formants_freqs_hz-1000)+100) #good for spec based method
      # formants_bandwidth_hz = ((3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.0125*torch.relu(formants_freqs_hz-1000)+100))#good for spec based method
      formants_bandwidth_hz = 0.65*(0.00625*torch.relu(formants_freqs_hz)+375)
      # formants_bandwidth_hz = (2**(torch.tanh(self.formant_bandwitdh_slop))*0.001*torch.relu(formants_freqs_hz-4000*torch.sigmoid(self.formant_bandwitdh_thres))+375*2**(torch.tanh(self.formant_bandwitdh_bias)))
      # formants_bandwidth_hz = torch.exp(0.4*torch.tanh(self.conv_formants_bandwidth(x_formants))) * (0.00625*torch.relu(formants_freqs_hz-0)+375)
      # formants_bandwidth_hz = (100*(torch.tanh(self.conv_formants_bandwidth(x_formants))) + (0.035*torch.relu(formants_freqs_hz-950)+250)) if self.wavebased else ((3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.0125*torch.relu(formants_freqs_hz-1000)+100))#good for spec based method
    #   formants_bandwidth_hz = (100*(torch.tanh(self.conv_formants_bandwidth(x_formants))) + (0.035*torch.relu(formants_freqs_hz-950)+250)) if self.wavebased else ((torch.sigmoid(self.conv_formants_bandwidth(x_formants))+0.2) * (3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.075*torch.relu(formants_freqs_hz-1000)+100))#good for spec based method
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*torch.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.relu(formants_freqs_hz-1000)+50)
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * (0.075*3*torch.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+(2*torch.sigmoid(self.formant_bandwitdh_ratio)+1)*50)
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*torch.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+50)
      formants_bandwidth = bandwidth_mel(formants_freqs_hz,formants_bandwidth_hz,self.n_mels) 
    #   formants_bandwidth_upper = formants_freqs_hz+formants_bandwidth_hz/2
    #   formants_bandwidth_lower = torch.clamp(formants_freqs_hz-formants_bandwidth_hz/2,min=1)
      # formants_bandwidth = (mel_scale(self.n_mels,formants_bandwidth_upper) - mel_scale(self.n_mels,formants_bandwidth_lower))/(self.n_mels*1.0)
      # formants_amplitude = F.softmax(torch.cumsum(-F.relu(self.conv_formants_amplitude(x_formants)),dim=1),dim=1)
      formants_amplitude_logit = self.conv_formants_amplitude(x_formants)
      formants_amplitude = F.softmax(formants_amplitude_logit,dim=1)

      formants_freqs_noise = torch.sigmoid(self.conv_formants_freqs_noise(x_formants))
    # #  relative freq:
    #   formants_freqs_hz = formants_freqs*(self.formant_freq_limits_diff[:,:self.n_formants]-self.formant_freq_limits_diff_low[:,:self.n_formants])+self.formant_freq_limits_diff_low[:,:self.n_formants]
    # #   formants_freqs_hz = formants_freqs*6839
    #   formants_freqs_hz = torch.cumsum(formants_freqs_hz,dim=1)

      # abs freq:
      formants_freqs_hz_noise = formants_freqs_noise*(self.formant_freq_limits_abs_noise[:,:self.n_formants_noise]-self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise])+self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise]
      if duomask:
         formants_freqs_hz_noise = torch.cat([formants_freqs_hz,formants_freqs_hz_noise],dim=1)
    #   formants_freqs_hz = formants_freqs*6839
      formants_freqs_noise = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz_noise)/(self.n_mels*1.0),min=0)
      
      # formants_freqs = formants_freqs + f0
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) *6839
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 150
      # formants_bandwidth_hz_noise = torch.sigmoid(self.conv_formants_bandwidth_noise(x_formants)) * 8000 + 2000
      formants_bandwidth_hz_noise = self.conv_formants_bandwidth_noise(x_formants)
      # formants_bandwidth_hz_noise_1 = F.softplus(formants_bandwidth_hz_noise[:,:1]) * 8000 + 2000 #2000-10000
      # formants_bandwidth_hz_noise_2 = torch.sigmoid(formants_bandwidth_hz_noise[:,1:]) * 2000 #0-2000
      formants_bandwidth_hz_noise_1 = F.softplus(formants_bandwidth_hz_noise[:,:1]) * 2344 + 586 #2000-10000
      formants_bandwidth_hz_noise_2 = torch.sigmoid(formants_bandwidth_hz_noise[:,1:]) * 586 #0-2000
      formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz_noise_1,formants_bandwidth_hz_noise_2],dim=1)
      if duomask:
         formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz,formants_bandwidth_hz_noise],dim=1)
      # formants_bandwidth_hz_noise = torch.sigmoid(self.conv_formants_bandwidth_noise(x_formants)) * 4000 + 1000
      # formants_bandwidth_hz_noise = torch.sigmoid(self.conv_formants_bandwidth_noise(x_formants)) * 4000
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*torch.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.relu(formants_freqs_hz-1000)+50)
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * (0.075*3*torch.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+(2*torch.sigmoid(self.formant_bandwitdh_ratio)+1)*50)
      # formants_bandwidth_hz = torch.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*torch.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+50)
      formants_bandwidth_noise = bandwidth_mel(formants_freqs_hz_noise,formants_bandwidth_hz_noise,self.n_mels)
    #   formants_bandwidth_upper = formants_freqs_hz+formants_bandwidth_hz/2
    #   formants_bandwidth_lower = torch.clamp(formants_freqs_hz-formants_bandwidth_hz/2,min=1)
    #   formants_bandwidth = (mel_scale(self.n_mels,formants_bandwidth_upper) - mel_scale(self.n_mels,formants_bandwidth_lower))/(self.n_mels*1.0)
      formants_amplitude_noise_logit = self.conv_formants_amplitude_noise(x_formants)
      if duomask:
         formants_amplitude_noise_logit = torch.cat([formants_amplitude_logit,formants_amplitude_noise_logit],dim=1)
      formants_amplitude_noise = F.softmax(formants_amplitude_noise_logit,dim=1)

      components = { 'f0':f0,
                     'f0_hz':f0_hz,
                     'loudness':loudness,
                     'amplitudes':amplitudes,
                     'amplitudes_h':amplitudes_h,
                     'freq_formants_hamon':formants_freqs,
                     'bandwidth_formants_hamon':formants_bandwidth,
                     'freq_formants_hamon_hz':formants_freqs_hz,
                     'bandwidth_formants_hamon_hz':formants_bandwidth_hz,
                     'amplitude_formants_hamon':formants_amplitude,
                     'freq_formants_noise':formants_freqs_noise,
                     'bandwidth_formants_noise':formants_bandwidth_noise,
                     'freq_formants_noise_hz':formants_freqs_hz_noise,
                     'bandwidth_formants_noise_hz':formants_bandwidth_hz_noise,
                     'amplitude_formants_noise':formants_amplitude_noise,
      }
      return components

class FromECoG(nn.Module):
    def __init__(self, outputs,residual=False,shape='3D'):
        super().__init__()
        self.residual=residual
        if shape =='3D':
            self.from_ecog = ln.Conv3d(1, outputs, [9,1,1], 1, [4,0,0])
        else:
            self.from_ecog = ln.Conv2d(1, outputs, [9,1], 1, [4,0])

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x

class ECoGMappingBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,fused_scale=True,residual=False,resample=[],pool=None,shape='3D'):
        super(ECoGMappingBlock, self).__init__()
        self.residual = residual
        self.pool = pool
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-1)//2)
        if shape=='2D':
            conv=ln.Conv2d
            maxpool = nn.MaxPool2d
            avgpool = nn.AvgPool2d
        if shape=='3D':
            conv=ln.Conv3d
            maxpool = nn.MaxPool3d
            avgpool = nn.AvgPool3d
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = nn.GroupNorm(min(inputs,32),inputs)
        else:
            self.norm1 = nn.GroupNorm(min(outputs,32),outputs)
        if pool is None:
            self.conv1 = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
        else:
            self.conv1 = conv(inputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
            self.pool1 = maxpool(self.resample,self.resample) if self.pool=='Max' else avgpool(self.resample,self.resample)
        if self.inputs_resample or self.dim_missmatch:
            if pool is None:
                self.convskip = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
            else:
                self.convskip = conv(inputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
                self.poolskip = maxpool(self.resample,self.resample) if self.pool=='Max' else avgpool(self.resample,self.resample)
                
        self.conv2 = conv(outputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(outputs,32),outputs)

    def forward(self,x):
        if self.residual:
            x = F.leaky_relu(self.norm1(x),0.2)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
                if self.pool is not None:
                    x_skip = self.poolskip(x_skip)
            else:
                x_skip = x
            x = F.leaky_relu(self.norm2(self.conv1(x)),0.2)
            if self.pool is not None:
                x = self.poolskip(x)
            x = self.conv2(x)
            x = x_skip + x
        else:
            x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
            x = F.leaky_relu(self.norm2(self.conv2(x)),0.2)
        return x



@ECOG_ENCODER.register("ECoGMappingBottleneck")
class ECoGMapping_Bottleneck(nn.Module):
    def __init__(self,n_mels,n_formants,n_formants_noise=1):
         super(ECoGMapping_Bottleneck, self).__init__()
         self.n_formants = n_formants
         self.n_mels = n_mels
         self.n_formants_noise = n_formants_noise

         self.formant_freq_limits_diff = torch.tensor([950.,2450.,2100.]).reshape([1,3,1]) #freq difference
         self.formant_freq_limits_diff_low = torch.tensor([300.,300.,0.]).reshape([1,3,1]) #freq difference

         # self.formant_freq_limits_abs = torch.tensor([950.,2800.,3400.,4700.]).reshape([1,4,1]) #freq difference
         # self.formant_freq_limits_abs_low = torch.tensor([300.,600.,2800.,3400]).reshape([1,4,1]) #freq difference

         # self.formant_freq_limits_abs = torch.tensor([950.,3300.,3600.,4700.]).reshape([1,4,1]) #freq difference
         # self.formant_freq_limits_abs_low = torch.tensor([300.,600.,2700.,3400]).reshape([1,4,1]) #freq difference

         self.formant_freq_limits_abs = torch.tensor([950.,3400.,3800.,5000.,6000.,7000.]).reshape([1,6,1]) #freq difference
         self.formant_freq_limits_abs_low = torch.tensor([300.,700.,1800.,3400,5000.,6000.]).reshape([1,6,1]) #freq difference

         # self.formant_freq_limits_abs_noise = torch.tensor([7000.]).reshape([1,1,1]) #freq difference
         self.formant_freq_limits_abs_noise = torch.tensor([8000.,7000.,7000.]).reshape([1,3,1]) #freq difference
         self.formant_freq_limits_abs_noise_low = torch.tensor([4000.,3000.,3000.]).reshape([1,3,1]) #freq difference

         self.formant_bandwitdh_ratio = Parameter(torch.Tensor(1))
         self.formant_bandwitdh_slop = Parameter(torch.Tensor(1))
         with torch.no_grad():
            nn.init.constant_(self.formant_bandwitdh_ratio,0)
            nn.init.constant_(self.formant_bandwitdh_slop,0)

         
         self.from_ecog = FromECoG(16,residual=True)
         self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,resample = [2,1,1],pool='MAX')
         self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,resample = [2,1,1],pool='MAX')
         self.norm_mask = nn.GroupNorm(32,64) 
         self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
         self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,resample = [2,2,2],pool='MAX')
         self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,resample = [2,2,2],pool='MAX')
         self.norm = nn.GroupNorm(32,256)
         self.conv5 = ln.Conv1d(256,256,3,1,1)
         self.norm2 = nn.GroupNorm(32,256)
         self.conv6 = ln.ConvTranspose1d(256, 128, 3, 2, 1, transform_kernel=True)
         self.norm3 = nn.GroupNorm(32,128)
         self.conv7 = ln.ConvTranspose1d(128, 64, 3, 2, 1, transform_kernel=True)
         self.norm4 = nn.GroupNorm(32,64)
         self.conv8 = ln.ConvTranspose1d(64, 32, 3, 2, 1, transform_kernel=True)
         self.norm5 = nn.GroupNorm(32,32)
         self.conv9 = ln.ConvTranspose1d(32, 32, 3, 2, 1, transform_kernel=True)
         self.norm6 = nn.GroupNorm(32,32)

         self.conv_fundementals = ln.Conv1d(32,32,3,1,1)
         self.norm_fundementals = nn.GroupNorm(32,32)
         self.f0_drop = nn.Dropout()
         self.conv_f0 = ln.Conv1d(32,1,1,1,0)
         self.conv_amplitudes = ln.Conv1d(32,2,1,1,0)
         self.conv_amplitudes_h = ln.Conv1d(32,2,1,1,0)
         self.conv_loudness = ln.Conv1d(32,1,1,1,0,bias_initial=-9.)

         self.conv_formants = ln.Conv1d(32,32,3,1,1)
         self.norm_formants = nn.GroupNorm(32,32)
         self.conv_formants_freqs = ln.Conv1d(32,n_formants,1,1,0)
         self.conv_formants_bandwidth = ln.Conv1d(32,n_formants,1,1,0)
         self.conv_formants_amplitude = ln.Conv1d(32,n_formants,1,1,0)

         self.conv_formants_freqs_noise = ln.Conv1d(32,n_formants_noise,1,1,0)
         self.conv_formants_bandwidth_noise = ln.Conv1d(32,n_formants_noise,1,1,0)
         self.conv_formants_amplitude_noise = ln.Conv1d(32,n_formants_noise,1,1,0)
      

    def forward(self,ecog,mask_prior,mni):
         x_common_all = []
         for d in range(len(ecog)):
            x = ecog[d]
            x = x.reshape([-1,1,x.shape[1],15,15])
            mask_prior_d = mask_prior[d].reshape(-1,1,1,15,15)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
            mask = mask[:,:,4:]
            if mask_prior is not None:
                mask = mask*mask_prior_d
            x = x[:,:,4:]
            x = x*mask
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.max(-1)[0].max(-1)[0]
            x = self.conv5(F.leaky_relu(self.norm(x),0.2))
            x = self.conv6(F.leaky_relu(self.norm2(x),0.2))
            x = self.conv7(F.leaky_relu(self.norm3(x),0.2))
            x = self.conv8(F.leaky_relu(self.norm4(x),0.2))
            x = self.conv9(F.leaky_relu(self.norm5(x),0.2))
            x_common = F.leaky_relu(self.norm6(x),0.2)
            x_common_all += [x_common]

         x_common = torch.cat(x_common_all,dim=0)
         loudness = F.softplus(self.conv_loudness(x_common))
         amplitudes = F.softmax(self.conv_amplitudes(x_common),dim=1)
         amplitudes_h = F.softmax(self.conv_amplitudes_h(x_common),dim=1)

         # x_fundementals = F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2)
         x_fundementals = self.f0_drop(F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2))
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals))
         # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
         
         # f0 in hz:
         # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * 528 + 88 # 88hz < f0 < 616 hz
         f0_hz = torch.sigmoid(self.conv_f0(x_fundementals)) * 332 + 88 # 88hz < f0 < 420 hz
         f0 = torch.clamp(mel_scale(self.n_mels,f0_hz)/(self.n_mels*1.0),min=0.0001)

         x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
         formants_freqs = torch.sigmoid(self.conv_formants_freqs(x_formants))
         # formants_freqs = torch.cumsum(formants_freqs,dim=1)
         # formants_freqs = formants_freqs

         # abs freq
         formants_freqs_hz = formants_freqs*(self.formant_freq_limits_abs[:,:self.n_formants]-self.formant_freq_limits_abs_low[:,:self.n_formants])+self.formant_freq_limits_abs_low[:,:self.n_formants]
         formants_freqs = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz)/(self.n_mels*1.0),min=0)

         # formants_freqs = formants_freqs + f0
         # formants_bandwidth = torch.sigmoid(self.conv_formants_bandwidth(x_formants))
         # formants_bandwidth_hz = (3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.075*torch.relu(formants_freqs_hz-1000)+100)
         formants_bandwidth_hz = 0.65*(0.00625*torch.relu(formants_freqs_hz)+375)
         # formants_bandwidth_hz = (torch.sigmoid(self.conv_formants_bandwidth(x_formants))) * (3*torch.sigmoid(self.formant_bandwitdh_ratio))*(0.075*torch.relu(formants_freqs_hz-1000)+100)
         formants_bandwidth = bandwidth_mel(formants_freqs_hz,formants_bandwidth_hz,self.n_mels)
         formants_amplitude_logit = self.conv_formants_amplitude(x_formants)
         formants_amplitude = F.softmax(formants_amplitude_logit,dim=1)

         formants_freqs_noise = torch.sigmoid(self.conv_formants_freqs_noise(x_formants))
         formants_freqs_hz_noise = formants_freqs_noise*(self.formant_freq_limits_abs_noise[:,:self.n_formants_noise]-self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise])+self.formant_freq_limits_abs_noise_low[:,:self.n_formants_noise]
         # formants_freqs_hz_noise = formants_freqs_noise*(self.formant_freq_limits_abs_noise[:,:1]-self.formant_freq_limits_abs_noise_low[:,:1])+self.formant_freq_limits_abs_noise_low[:,:1]
         formants_freqs_hz_noise = torch.cat([formants_freqs_hz,formants_freqs_hz_noise],dim=1)
         formants_freqs_noise = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz_noise)/(self.n_mels*1.0),min=0)
         # formants_bandwidth_hz_noise = F.relu(self.conv_formants_bandwidth_noise(x_formants)) * 8000 + 2000
         # formants_bandwidth_noise = bandwidth_mel(formants_freqs_hz_noise,formants_bandwidth_hz_noise,self.n_mels)
         # formants_amplitude_noise = F.softmax(self.conv_formants_amplitude_noise(x_formants),dim=1)
         formants_bandwidth_hz_noise = self.conv_formants_bandwidth_noise(x_formants)
         formants_bandwidth_hz_noise_1 = F.softplus(formants_bandwidth_hz_noise[:,:1]) * 2344 + 586 #2000-10000
         formants_bandwidth_hz_noise_2 = torch.sigmoid(formants_bandwidth_hz_noise[:,1:]) * 586 #0-2000
         formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz_noise_1,formants_bandwidth_hz_noise_2],dim=1)
         formants_bandwidth_hz_noise = torch.cat([formants_bandwidth_hz,formants_bandwidth_hz_noise],dim=1)
         formants_bandwidth_noise = bandwidth_mel(formants_freqs_hz_noise,formants_bandwidth_hz_noise,self.n_mels)
         formants_amplitude_noise_logit = self.conv_formants_amplitude_noise(x_formants)
         formants_amplitude_noise_logit = torch.cat([formants_amplitude_logit,formants_amplitude_noise_logit],dim=1)
         formants_amplitude_noise = F.softmax(formants_amplitude_noise_logit,dim=1)

         components = { 'f0':f0,
                        'f0_hz':f0_hz,
                        'loudness':loudness,
                        'amplitudes':amplitudes,
                        'amplitudes_h':amplitudes_h,
                        'freq_formants_hamon':formants_freqs,
                        'bandwidth_formants_hamon':formants_bandwidth,
                        'freq_formants_hamon_hz':formants_freqs_hz,
                        'bandwidth_formants_hamon_hz':formants_bandwidth_hz,
                        'amplitude_formants_hamon':formants_amplitude,
                        'freq_formants_noise':formants_freqs_noise,
                        'bandwidth_formants_noise':formants_bandwidth_noise,
                        'freq_formants_noise_hz':formants_freqs_hz_noise,
                        'bandwidth_formants_noise_hz':formants_bandwidth_hz_noise,
                        'amplitude_formants_noise':formants_amplitude_noise,
         }
         return components


class BackBone(nn.Module):
    def __init__(self,attentional_mask=True):
        super(BackBone, self).__init__()
        self.attentional_mask = attentional_mask
        self.from_ecog = FromECoG(16,residual=True,shape='2D')
        self.conv1 = ECoGMappingBlock(16,32,[5,1],residual=True,resample = [1,1],shape='2D')
        self.conv2 = ECoGMappingBlock(32,64,[3,1],residual=True,resample = [1,1],shape='2D')
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv2d(64,1,[3,1],1,[1,0])
    
    def forward(self,ecog):
        x_common_all = []
        mask_all=[]
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.unsqueeze(1)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            if self.attentional_mask:
                mask = F.relu(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
                mask = mask[:,:,16:]
                x = x[:,:,16:]
                mask_all +=[mask]
            else:
                # mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
                # mask = mask[:,:,16:]
                x = x[:,:,16:]
                # x = x*mask
                
            x_common_all +=[x]
            
        x_common = torch.cat(x_common_all,dim=0)
        if self.attentional_mask:
            mask = torch.cat(mask_all,dim=0)
        return x_common,mask.squeeze(1) if self.attentional_mask else None

class ECoGEncoderFormantHeads(nn.Module):
    def __init__(self,inputs,n_mels,n_formants):
        super(ECoGEncoderFormantHeads,self).__init__()
        self.n_mels = n_mels
        self.f0 = ln.Conv1d(inputs,1,1)
        self.loudness = ln.Conv1d(inputs,1,1)
        self.amplitudes = ln.Conv1d(inputs,2,1)
        self.freq_formants = ln.Conv1d(inputs,n_formants,1)
        self.bandwidth_formants = ln.Conv1d(inputs,n_formants,1)
        self.amplitude_formants = ln.Conv1d(inputs,n_formants,1)

    def forward(self,x):
        loudness = F.relu(self.loudness(x))
        f0 = torch.sigmoid(self.f0(x)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
        # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
        # f0 = torch.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
        amplitudes = F.softmax(self.amplitudes(x),dim=1)
        freq_formants = torch.sigmoid(self.freq_formants(x))
        freq_formants = torch.cumsum(freq_formants,dim=1)
        bandwidth_formants = torch.sigmoid(self.bandwidth_formants(x))
        amplitude_formants = F.softmax(self.amplitude_formants(x),dim=1)
        return {'f0':f0,
                'loudness':loudness,
                'amplitudes':amplitudes,
                'freq_formants':freq_formants,
                'bandwidth_formants':bandwidth_formants,
                'amplitude_formants':amplitude_formants,}

@ECOG_ENCODER.register("ECoGMappingTransformer")
class ECoGMapping_Transformer(nn.Module):
    def __init__(self,n_mels,n_formants,SeqLen=128,hidden_dim=256,dim_feedforward=256,encoder_only=False,attentional_mask=False,n_heads=1,non_local=False):
        super(ECoGMapping_Transformer, self).__init__()
        self.n_mels = n_mels,
        self.n_formant = n_formants,
        self.encoder_only = encoder_only,
        self.attentional_mask = attentional_mask,
        self.backbone = BackBone(attentional_mask=attentional_mask)
        self.position_encoding = build_position_encoding(SeqLen,hidden_dim,'MNI')
        self.input_proj = ln.Conv2d(64, hidden_dim, kernel_size=1)
        if non_local:
            Transformer = TransformerNL
        else:
            Transformer = TransformerTS
        self.transformer = Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=6,
                                    num_decoder_layers=6, dim_feedforward=dim_feedforward, dropout=0.1,
                                    activation="relu", normalize_before=False,
                                    return_intermediate_dec=False,encoder_only = encoder_only)
        self.output_proj = ECoGEncoderFormantHeads(hidden_dim,n_mels,n_formants)
        self.query_embed = nn.Embedding(SeqLen, hidden_dim)
    
    def forward(self,x,mask_prior,mni):
        features,mask = self.backbone(x)
        pos = self.position_encoding(mni)
        hs = self.transformer(self.input_proj(features), mask if self.attentional_mask else None, self.query_embed.weight, pos)
        if not self.encoder_only:
            hs,encoded = hs
            out = self.output_proj(hs)
        else:
            _,encoded = hs
            encoded = encoded.max(-1)[0]
            out = self.output_proj(encoded)
        return out
        


