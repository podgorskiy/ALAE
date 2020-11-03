import pdb
import torch
from torch import nn
# from torch.nn import functional as F
# from registry import *
import lreq as ln
import json
from tqdm import tqdm
import os
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.nn.parameter import Parameter
from custom_adam import LREQAdam
from ECoGDataSet import ECoGDataset
from net_formant import mel_scale, hz2ind
import matplotlib.pyplot as plt
# from matplotlib.pyplot import ion; ion()
import scipy.signal
import scipy.io.wavfile
import math
from net_formant import amplitude
import torchaudio
def spsi(msgram, fftsize, hop_length) :
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """
    msgram = np.sqrt(msgram)
    numBins, numFrames  = msgram.shape
    y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
            m_mag=msgram[:, i] 
            for j in range(1,numBins-1) : 
                if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                    alpha=m_mag[j-1];
                    beta=m_mag[j];
                    gamma=m_mag[j+1];
                    denom=alpha-2*beta+gamma;
                    
                    if(denom!=0) :
                        p=0.5*(alpha-gamma)/denom;
                    else :
                        p=0;
                        
                    #phaseRate=2*math.pi*(j-1+p)/fftsize;    #adjusted phase rate
                    phaseRate=2*math.pi*(j+p)/fftsize;    #adjusted phase rate
                    m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                    peakPhase=m_phase[j];
                    
                    # If actual peak is to the right of the bin freq
                    if (p>0) :
                        # First bin to right has pi shift
                        bin=j+1;
                        m_phase[bin]=peakPhase+math.pi;
                        
                        # Bins to left have shift of pi
                        bin=j-1;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                            m_phase[bin]=peakPhase+math.pi;
                            bin=bin-1;
                        
                        #Bins to the right (beyond the first) have 0 shift
                        bin=j+2;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase;
                            bin=bin+1;
                            
                    #if actual peak is to the left of the bin frequency
                    if(p<0) :
                        # First bin to left has pi shift
                        bin=j-1;
                        m_phase[bin]=peakPhase+math.pi;

                        # and bins to the right of me - here I am stuck in the middle with you
                        bin=j+1;
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase+math.pi;
                            bin=bin+1;
                        
                        # and further to the left have zero shift
                        bin=j-2;
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                            m_phase[bin]=peakPhase;
                            bin=bin-1;
                            
                #end ops for peaks
            #end loop over fft bins with

            magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
            magphase[0]=0; magphase[numBins-1] = 0 #remove dc and nyquist
            m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
            
            #overlap and add
            m_recon=np.real(np.fft.ifft(m_recon))*m_win
            y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    return y_out

def freq_coloring(sample,ind,color='r'):
   for b in range(sample.shape[0]):
      for t in range(sample.shape[2]):
         if color == 'r':
            sample[b,0,t,ind[b,0,t]]=1
         if color == 'g':
            sample[b,1,t,ind[b,0,t]]=1
         if color == 'b':
            sample[b,2,t,ind[b,0,t]]=1
         if color == 'y':
            sample[b,0,t,ind[b,0,t]]=1;sample[b,1,t,ind[b,0,t]]=1
         if color == 'c':
            sample[b,1,t,ind[b,0,t]]=1;sample[b,2,t,ind[b,0,t]]=1
         if color == 'm':
            sample[b,0,t,ind[b,0,t]]=1;sample[b,2,t,ind[b,0,t]]=1
   return sample

def voicing_coloring(sample,amplitude):
   for b in range(sample.shape[0]):
      for t in range(sample.shape[2]):
         sample[b,:,t]=sample[b,0:1,t]*(amplitude[b,0,t]*torch.tensor([0.,0.35,0.67]) + amplitude[b,1,t]*torch.tensor([1.0,0.87,0.])).unsqueeze(dim=-1) # voicing for blue, unvoicing for yellow
   return sample
         
def color_spec(spec,components,n_mels):
   clrs = ['g','y','b','m','c']
   # sample_in = spec.repeat(1,3,1,1)
   # sample_in = sample_in * 0.5 + 0.5
   f0=(components['f0']*n_mels).int().clamp(min=0,max=n_mels-1)
   formants_freqs=(components['freq_formants_hamon']*n_mels).int().clamp(min=0,max=n_mels-1)
   sample_in = spec
   sample_in_color_voicing = sample_in.clone()
   sample_in_color_voicing = voicing_coloring(sample_in_color_voicing,components['amplitudes'])
   sample_in_color_hamon_voicing = sample_in.clone()
   sample_in_color_hamon_voicing = voicing_coloring(sample_in_color_hamon_voicing,components['amplitudes_h'])
   # sample_in_color_freq = sample_in.clone()
   # sample_in_color_freq = sample_in_color_freq/2
   # sample_in_color_freq = freq_coloring(sample_in_color_freq,f0,'r')
   # for j in range(formants_freqs.shape[1]):
   #    sample_in_color_freq = freq_coloring(sample_in_color_freq,formants_freqs[:,j].unsqueeze(1),clrs[j])
   return sample_in_color_voicing,sample_in_color_hamon_voicing

def subfigure_plot(ax,spec,components,n_mels,which_formant='hamon',formant_line=True,title=None,linear=False,n_fft=256,power=False,ecog=None):
   clrs = ['tab:green','tab:orange','tab:blue','tab:purple','tab:pink','tab:brown','tab:cyan']
   
   if formant_line:

      ax.imshow(np.clip(1-spec.detach().cpu().numpy().squeeze().T,0,1),vmin=0.0,vmax=1.0)
      f0=(components['f0']*n_mels).clamp(min=0,max=n_mels-1)
      f0_hz=(components['f0_hz']).clamp(min=0)
      formants_freqs=(components['freq_formants_'+which_formant]*n_mels).clamp(min=0,max=n_mels-1)
      formants_freqs_hz=(components['freq_formants_'+which_formant+'_hz'])
      if ecog is not None:
         f0_ecog=(ecog['f0']*n_mels).clamp(min=0,max=n_mels-1)
         f0_hz_ecog=(ecog['f0_hz']).clamp(min=0)

      if linear:
         f0 = hz2ind(f0_hz,n_fft).clamp(min=0,max=n_fft-1)
         if ecog is not None:
            f0_ecog = hz2ind(f0_hz_ecog,n_fft).clamp(min=0,max=n_fft-1)
      ax.plot(f0.squeeze().detach().cpu().numpy(),color='tab:red',linewidth=1,label='f0')
      for i in range(formants_freqs.shape[1]):
         alpha = components['amplitude_formants_'+which_formant][:,i].squeeze().detach().cpu().numpy()
         if linear:
            ax.plot(hz2ind(formants_freqs_hz[:,i],n_fft).clamp(min=0,max=n_fft-1).squeeze().detach().cpu().numpy(),color=clrs[i],linewidth=2,label='f'+str(i+1))
            minimum = hz2ind(formants_freqs_hz[:,i] - components['bandwidth_formants_'+which_formant+'_hz'][:,i]/2,n_fft).clamp(min=0).squeeze().detach().cpu().numpy()
            maximum = hz2ind(formants_freqs_hz[:,i] + components['bandwidth_formants_'+which_formant+'_hz'][:,i]/2,n_fft).clamp(max=n_fft-1).squeeze().detach().cpu().numpy()
         else:
            ax.plot(formants_freqs[:,i].squeeze().detach().cpu().numpy(),color=clrs[i],linewidth=1,label='f'+str(i+1))
            minimum = mel_scale(n_mels,(formants_freqs_hz[:,i] - components['bandwidth_formants_'+which_formant+'_hz'][:,i]/2).clamp(min=0.1)).clamp(min=0).squeeze().detach().cpu().numpy()
            maximum = mel_scale(n_mels,formants_freqs_hz[:,i] + components['bandwidth_formants_'+which_formant+'_hz'][:,i]/2).clamp(max=n_mels-1).squeeze().detach().cpu().numpy()
            # minimum = (formants_freqs[:,i] - components['bandwidth_formants_'+which_formant][:,i]/2).squeeze().detach().cpu().numpy()
            # maximum = (formants_freqs[:,i] + components['bandwidth_formants_'+which_formant][:,i]/2).squeeze().detach().cpu().numpy()
         ax.fill_between(range(minimum.shape[0]),minimum,maximum,where=(minimum<=maximum),color=clrs[i],alpha=0.2) 
      if ecog is not None:
         for i in range(ecog['freq_formants_'+which_formant+'_hz'].shape[1]):
            ax.plot(hz2ind(ecog['freq_formants_'+which_formant+'_hz'][:,i],n_fft).clamp(min=0,max=n_fft-1).squeeze().detach().cpu().numpy(),color=clrs[i],linewidth=2,label='f'+str(i+1)+'_ecog',linestyle='--')
      ax.legend()
   else:
      if title=='loudness':
         loudness = torchaudio.transforms.AmplitudeToDB()(components['loudness'])
         if ecog is not None:
            loudness_ecog = torchaudio.transforms.AmplitudeToDB()(ecog['loudness'])
         loudness = (loudness+100)*2
         plt.sca(ax)
         plt.yticks(range(0,200,20),(np.arange(0,200,20)/2-100).astype(str))
         ax.imshow(np.clip(1-spec.detach().cpu().numpy().squeeze().T,0,1),vmin=0.0,vmax=1.0)
         ax.plot(loudness.squeeze().detach().cpu().numpy(),color='r',linewidth=2,label='loudness')
         if ecog is not None:
            loudness_ecog = (loudness_ecog+100)*2
            ax.plot(loudness_ecog.squeeze().detach().cpu().numpy(),color='r',linewidth=2,label='loudness_ecog',linestyle='--')
         ax.legend()
      elif title == 'amplitude_hamon' or title == 'amplitude_noise':
         comp = components['amplitude_formants_hamon'] if title=='amplitude_hamon' else components['amplitude_formants_noise']
         if ecog is not None:
            comp_ecog = ecog['amplitude_formants_hamon'] if title=='amplitude_hamon' else ecog['amplitude_formants_noise']
         plt.sca(ax)
         ax.set_yticks(np.arange(0,200,20))
         ax.set_yticklabels(np.arange(0,200,20)/200)
         #plt.yticks(range(0,200,20),(np.arange(0,1,20)).astype(str))
         ax.imshow(np.clip(1-spec.detach().cpu().numpy().squeeze().T,0,1),vmin=0.0,vmax=1.0)
         for i in range(comp.shape[1]):
            ax.plot(200*comp[:,i].squeeze().detach().cpu().numpy().T,linewidth=2,color=clrs[i])
         if ecog is not None:
            for i in range(comp_ecog.shape[1]):
               ax.plot(200*comp_ecog[:,i].squeeze().detach().cpu().numpy().T,linewidth=2,color=clrs[i],linestyle='--')
         # ax.plot(((torchaudio.transforms.AmplitudeToDB()(comp)+100)*2).squeeze().detach().cpu().numpy().T,linewidth=2,linestyle='--')
      else:
         ax.imshow(np.clip(spec.detach().cpu().numpy().squeeze().T,0,1),vmin=0.0,vmax=1.0)
   ax.set_ylim(0,n_fft if linear else n_mels)
   if title is not None:
      ax.set_title(title)


def save_sample(sample,ecog,mask_prior,mni,encoder,decoder,ecog_encoder,epoch,label,x_denoise=None,decoder_mel=None,x_mel=None,mode='test',path='training_artifacts/formantsysth_voicingandunvoicing_loudness',tracker=None,linear=False,n_fft=256,duomask=True,x_amp=None):
   os.makedirs(path, exist_ok=True)
   labels = ()
   # import pdb; pdb.set_trace()
   tracker.register_means(epoch)
   for i in range(len(label)):
      labels += label[i]
   with torch.no_grad():
      encoder.eval()
      decoder.eval()
      if decoder_mel is not None:
         decoder_mel.eval()
      if ecog_encoder is not None:
         ecog_encoder.eval()
      sample_in_all = torch.tensor([])
      sample_in_color_freq_all = torch.tensor([])
      sample_in_color_voicing_all = torch.tensor([])
      rec_all = torch.tensor([])
      rec_denoise_all = torch.tensor([])
      rec_denoise_wave_all = torch.tensor([])
      components_all = []
      
      if ecog_encoder is not None:
         sample_in_color_freq_ecog_all = torch.tensor([])
         sample_in_color_voicing_ecog_all = torch.tensor([])
         rec_ecog_all = torch.tensor([])
         rec_denoise_ecog_all = torch.tensor([])
         rec_denoise_ecog_wave_all = torch.tensor([])
         components_ecog_all = []
      n_mels = sample.shape[-1]
      if decoder_mel is None:
         fig,axs = plt.subplots(9,sample.shape[0],figsize=(5*sample.shape[0],23*(4 if linear else 1))) if ecog_encoder is None else plt.subplots(11,sample.shape[0],figsize=(5*sample.shape[0],28*4))
      else:
         fig,axs = plt.subplots(10,sample.shape[0],figsize=(5*sample.shape[0],27*(4 if linear else 1))) if ecog_encoder is None else plt.subplots(11,sample.shape[0],figsize=(5*sample.shape[0],28*4))
      loc = (hz2ind(np.array([250,500,1000,2000,4000,8000]),sample.shape[-1])) if linear else (mel_scale(sample.shape[-1],np.array([250,500,1000,2000,4000,8000]),pt=False))
      plt.setp(axs,yticks=loc,yticklabels=['250','500','1k','2k','4k','8k'])
      # import pdb; pdb.set_trace()
      for i in range(0,sample.shape[0],1):
         sample_in = sample[i:np.minimum(i+1,sample.shape[0])]
         sample_in_denoise = x_denoise[i:np.minimum(i+1,sample.shape[0])] if x_denoise is not None else None
         x_amp_in = x_amp[i:np.minimum(i+1,sample.shape[0])] if x_amp is not None else None
         if ecog_encoder is not None:
            ecog_in = [ecog[j][i:np.minimum(i+1,sample.shape[0])] for j in range(len(ecog))]
            mask_prior_in = [mask_prior[j][i:np.minimum(i+1,sample.shape[0])] for j in range(len(ecog))]
            mni_in = mni[i:np.minimum(i+1,sample.shape[0])]
         components = encoder(sample_in,duomask=duomask,x_denoise=sample_in_denoise,noise_level = F.softplus(decoder.bgnoise_amp)*decoder.noise_dist.mean(),x_amp=x_amp_in)
         # components['amplitudes'][:,1:2] = torch.zeros(components['amplitudes'][:,0:1].shape)
         # components['freq_formants_hamon'] = components['freq_formants_hamon'][:,:2]
         # components['freq_formants_hamon_hz'] = components['freq_formants_hamon_hz'][:,:2]
         # components['bandwidth_formants_hamon'] = components['bandwidth_formants_hamon'][:,:2]
         # components['bandwidth_formants_hamon_hz'] = components['bandwidth_formants_hamon_hz'][:,:2]
         # components['amplitude_formants_hamon'] = components['amplitude_formants_hamon'][:,:2]
         rec = decoder(components)
         # import pdb; pdb.set_trace()
         decoder.return_wave = True
         rec_denoise,rec_denoise_wave = decoder(components,enable_bgnoise=False)
         decoder.return_wave = False
         
         if decoder_mel is not None:
            rec_mel = decoder_mel(components,enable_bgnoise=False)
            rec_mel = rec_mel.repeat(1,3,1,1)
            rec_mel = rec_mel * 0.5 + 0.5
         sample_in = sample_in.repeat(1,3,1,1)
         sample_in = sample_in * 0.5 + 0.5
         rec = rec.repeat(1,3,1,1)
         rec = rec * 0.5 + 0.5
         rec_denoise = rec_denoise.repeat(1,3,1,1)
         rec_denoise = rec_denoise * 0.5 + 0.5
         sample_in_all = torch.cat([sample_in_all,sample_in],dim=0)
         # sample_in_color_freq_all = torch.cat([sample_in_color_freq_all,sample_in_color_freq],dim=0)
         # sample_in_color_voicing_all = torch.cat([sample_in_color_voicing_all,sample_in_color_voicing],dim=0)
         rec_all = torch.cat([rec_all,rec],dim=0)
         rec_denoise_all = torch.cat([rec_denoise_all,rec_denoise],dim=0)
         rec_denoise_wave_all = torch.cat([rec_denoise_wave_all,rec_denoise_wave],dim=0)
         components_all += [components]
         if ecog_encoder is not None:
            components_ecog = ecog_encoder(ecog_in,mask_prior_in,mni=mni_in)
            rec_ecog = decoder(components_ecog)
            rec_ecog = rec_ecog.repeat(1,3,1,1)
            rec_ecog = rec_ecog * 0.5 + 0.5
            decoder.return_wave = True
            rec_ecog_denoise,rec_ecog_denoise_wave = decoder(components_ecog,enable_bgnoise=False)
            decoder.return_wave = False
            rec_ecog_denoise = rec_ecog_denoise.repeat(1,3,1,1)
            rec_ecog_denoise = rec_ecog_denoise * 0.5 + 0.5
            rec_ecog_all = torch.cat([rec_ecog_all,rec_ecog],dim=0)
            rec_denoise_ecog_all = torch.cat([rec_denoise_ecog_all,rec_ecog_denoise],dim=0)
            rec_denoise_ecog_wave_all = torch.cat([rec_denoise_ecog_wave_all,rec_ecog_denoise_wave],dim=0)
            components_ecog_all += [components_ecog]
      # if ecog_encoder is not None:
      #    mse = ((rec_ecog_all-sample_in_all)**2).mean(dim=[1,2,3])
      #    mse,indices = torch.sort(mse)
      #    indices=indices.detach().cpu().numpy().astype(int)
      #    sample_in_all = sample_in_all[indices]
      #    rec_all=rec_all[indices]
      #    rec_denoise_all=rec_denoise_all[indices]
      #    rec_denoise_wave_all=rec_denoise_wave_all[indices]
      #    components_all = np.array(components_all)[indices]
      #    rec_ecog_all = rec_ecog_all[indices]
      #    rec_denoise_ecog_all = rec_denoise_ecog_all[indices]
      #    rec_denoise_ecog_wave_all = rec_denoise_ecog_wave_all[indices]
      #    components_ecog_all = np.array(components_ecog_all)[indices]
      #    labels = np.array(labels)[indices]
      # else:
      #    indices = np.argsort(np.array(labels))
      #    sample_in_all = sample_in_all[indices]
      #    rec_all=rec_all[indices]
      #    rec_denoise_all=rec_denoise_all[indices]
      #    rec_denoise_wave_all=rec_denoise_wave_all[indices]
      #    components_all = np.array(components_all)[indices]
      #    labels = np.array(labels)[indices]

      for i in range(0,sample.shape[0],1):
         sample_in_color_voicing,sample_in_color_hamon_voicing = color_spec(sample_in_all[i:i+1],components_all[i],n_mels)
         subfigure_plot(axs[0,i],sample_in_all[i:i+1],components_all[i],n_mels,formant_line=False,title='\''+labels[i]+'\'',linear=linear,n_fft=n_fft)
         subfigure_plot(axs[1,i],sample_in_all[i:i+1],components_all[i],n_mels,formant_line=False,title='loudness',linear=linear,n_fft=n_fft,power=decoder.power_synth,ecog = None if ecog_encoder is None else components_ecog_all[i])
         subfigure_plot(axs[2,i],sample_in_all[i:i+1],components_all[i],n_mels,formant_line=True,title='formants_hamon',linear=linear,n_fft=n_fft,ecog = None if ecog_encoder is None else components_ecog_all[i])
         subfigure_plot(axs[3,i],sample_in_all[i:i+1],components_all[i],n_mels,which_formant='noise',formant_line=True,title='formants_noise',linear=linear,n_fft=n_fft,ecog = None if ecog_encoder is None else components_ecog_all[i])
         subfigure_plot(axs[4,i],sample_in_all[i:i+1],components_all[i],n_mels,formant_line=False,title='amplitude_hamon',linear=linear,n_fft=n_fft,ecog = None if ecog_encoder is None else components_ecog_all[i])
         subfigure_plot(axs[5,i],sample_in_all[i:i+1],components_all[i],n_mels,formant_line=False,title='amplitude_noise',linear=linear,n_fft=n_fft,ecog = None if ecog_encoder is None else components_ecog_all[i])
         subfigure_plot(axs[6,i],sample_in_color_voicing,components_all[i],n_mels,formant_line=False,title='alpha',linear=linear,n_fft=n_fft,ecog = None if ecog_encoder is None else components_ecog_all[i])
         # subfigure_plot(axs[4,i],sample_in_color_hamon_voicing,components_all[i],n_mels,formant_line=False,title='beta',linear=linear,n_fft=n_fft)
         subfigure_plot(axs[7,i],rec_all[i:i+1],components_all[i],n_mels,formant_line=False,title='rec',linear=linear,n_fft=n_fft)
         subfigure_plot(axs[8,i],rec_denoise_all[i:i+1],components_all[i],n_mels,formant_line=False,title='rec_denoise',linear=linear,n_fft=n_fft)
         if ecog_encoder is None and decoder_mel is not None:
            subfigure_plot(axs[9,i],rec_mel,components_all[i],rec_mel.shape[-1],formant_line=False,title='rec_mel',linear=False,n_fft=n_fft)

         if ecog_encoder is not None:

            # sample_in_color_voicing_ecog,sample_in_color_hamon_voicing_ecog = color_spec(sample_in_all[i:i+1],components_ecog_all[i],n_mels)
            # subfigure_plot(axs[9,i],sample_in_all[i:i+1],components_ecog_all[i],n_mels,formant_line=True,title='formants_ecog',linear=linear,n_fft=n_fft)
            # subfigure_plot(axs[10,i],sample_in_all[i:i+1],components_ecog_all[i],n_mels,which_formant='noise',formant_line=True,title='formants_noise_ecog',linear=linear,n_fft=n_fft)
            # subfigure_plot(axs[11,i],sample_in_all[i:i+1],components_ecog_all[i],n_mels,formant_line=False,title='loudness',linear=linear,n_fft=n_fft,power=decoder.power_synth)
            # subfigure_plot(axs[12,i],sample_in_all[i:i+1],components_ecog_all[i],n_mels,formant_line=False,title='amplitude_hamon',linear=linear,n_fft=n_fft)
            # subfigure_plot(axs[13,i],sample_in_all[i:i+1],components_ecog_all[i],n_mels,formant_line=False,title='amplitude_noise',linear=linear,n_fft=n_fft)
            # subfigure_plot(axs[14,i],sample_in_color_voicing_ecog,components_ecog_all[i],n_mels,formant_line=False,title='alpha_ecog',linear=linear,n_fft=n_fft)
            subfigure_plot(axs[9,i],rec_ecog_all[i:i+1],components_ecog_all[i],n_mels,formant_line=False,title='rec_ecog',linear=linear,n_fft=n_fft)
            subfigure_plot(axs[10,i],rec_denoise_ecog_all[i:i+1],components_ecog_all[i],n_mels,formant_line=False,title='rec_ecog_denoise',linear=linear,n_fft=n_fft)
            # sample_in_color_freq_ecog_all = torch.cat([sample_in_color_freq_ecog_all,sample_in_color_freq_ecog],dim=0)
            # sample_in_color_voicing_ecog_all = torch.cat([sample_in_color_voicing_ecog_all,sample_in_color_voicing_ecog],dim=0)
            
      # sample_in_all = sample_in_all.repeat(1,3,1,1)*0.5+0.5
      if ecog_encoder is not None:
         resultsample = torch.cat([sample_in_all, rec_all, rec_ecog_all], dim=0)
      else:
         resultsample = torch.cat([sample_in_all, rec_all], dim=0)
      resultsample = resultsample.transpose(-2,-1)
      # import pdb;pdb.set_trace()
      if mode == 'train':
         f = os.path.join(path,'result_train_%d.png' % (epoch + 1))
         f2 = os.path.join(path,'sample_train_%d.png' % (epoch + 1))
      if mode == 'test':
         f = os.path.join(path,'result_%d.png' % (epoch + 1))
         f2 = os.path.join(path,'sample_%d.png' % (epoch + 1))
      # import pdb;pdb.set_trace()
      
      fig.savefig(f, bbox_inches='tight',dpi=80)
      plt.close(fig)

      
      if linear:
         rec_all = amplitude(torch.cat((2*rec_all[:,0]-1).unbind(),0).transpose(-2,-1).detach().cpu().numpy(),-50,22.5,trim_noise=False)
         rec_wave = spsi(rec_all,(n_fft-1)*2,128)
         scipy.io.wavfile.write(f2+'.wav',16000,rec_wave)
         rec_denoise_all = amplitude(torch.cat((2*rec_denoise_all[:,0]-1).unbind(),0).transpose(-2,-1).detach().cpu().numpy(),-50,22.5,trim_noise=False)
         rec_wave_denoise = spsi(rec_denoise_all,(n_fft-1)*2,128)
         scipy.io.wavfile.write(f2+'denoise.wav',16000,rec_wave_denoise)
         if ecog_encoder is not None:
            rec_ecog_all = amplitude(torch.cat((2*rec_ecog_all[:,0]-1).unbind(),0).transpose(-2,-1).detach().cpu().numpy(),-50,22.5,trim_noise=False)
            rec_wave_ecog = spsi(rec_ecog_all,(n_fft-1)*2,128)
            scipy.io.wavfile.write(f2+'ecog.wav',16000,rec_wave_ecog)
            rec_denoise_ecog_all = amplitude(torch.cat((2*rec_denoise_ecog_all[:,0]-1).unbind(),0).transpose(-2,-1).detach().cpu().numpy(),-50,22.5,trim_noise=False)
            rec_wave_denoise_ecog = spsi(rec_denoise_ecog_all,(n_fft-1)*2,128)
            scipy.io.wavfile.write(f2+'denoiseecog.wav',16000,rec_wave_denoise_ecog)

      save_image(resultsample, f2, nrow=resultsample.shape[0]//(2 if ecog_encoder is None else 3))
      # import pdb;pdb.set_trace()
      if ecog_encoder is not None:
         scipy.io.wavfile.write(f2+'denoisewave.wav',16000,torch.cat(rec_denoise_wave_all.unbind(),1)[0].detach().cpu().numpy())
         scipy.io.wavfile.write(f2+'denoiseecogwave.wav',16000,torch.cat(rec_denoise_ecog_wave_all.unbind(),1)[0].detach().cpu().numpy())
         if mode =='test':
            return torch.cat(rec_denoise_ecog_wave_all.unbind(),1)[0].detach().cpu().numpy()
      else:
         scipy.io.wavfile.write(f2+'denoisewave.wav',16000,torch.cat(rec_denoise_wave_all.unbind(),1)[0].detach().cpu().numpy())
         if mode =='test':
            return torch.cat(rec_denoise_wave_all.unbind(),1)[0].detach().cpu().numpy()
  
      
      

def main():
   OUTPUT_DIR = 'training_artifacts/formantsysth_voicingandunvoicing_loudness_NY742'
   LOAD_DIR = ''
   # LOAD_DIR = 'training_artifacts/formantsysth_voicingandunvoicing_loudness_'
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   device = torch.device("cuda:0")
   encoder = FormantEncoder(n_mels=64,n_formants=2,k=30)
   decoder = FormantSysth(n_mels=64,k=30)
   encoder.cuda()
   decoder.cuda()
   if LOAD_DIR is not '':
      encoder.load_state_dict(torch.load(os.path.join(LOAD_DIR,'encoder_60.pth')))
      decoder.load_state_dict(torch.load(os.path.join(LOAD_DIR,'decoder_60.pth')))
   optimizer = LREQAdam([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ], lr=0.01, weight_decay=0)
   with open('train_param.json','r') as rfile:
        param = json.load(rfile)
   dataset = torch.utils.data.DataLoader(ECoGDataset(['NY742'],mode='train'),batch_size=32,shuffle=True, drop_last=True)
   dataset_test = torch.utils.data.DataLoader(ECoGDataset(['NY742'],mode='test'),batch_size=50,shuffle=False, drop_last=False)
   sample_dict_test = next(iter(dataset_test))
   sample_spec_test = sample_dict_test['spkr_re_batch_all'].to('cuda').float()
   for epoch in range(60):
      encoder.train()
      decoder.train()
      for sample_dict_train in tqdm(iter(dataset)):
         x = sample_dict_train['spkr_re_batch_all'].to('cuda').float()
         optimizer.zero_grad()
         f0,loudness,amplitudes,formants_freqs,formants_bandwidth,formants_amplitude = encoder(x)
         x_rec = decoder(f0,loudness,amplitudes,formants_freqs,formants_bandwidth,formants_amplitude)
         loss = torch.mean((x-x_rec).abs())
         loss.backward()
         optimizer.step()
      save_sample(sample_spec_test,encoder,decoder,epoch,mode='test',path=OUTPUT_DIR)
      save_sample(x,encoder,decoder,epoch,mode='train',path=OUTPUT_DIR)
      torch.save(encoder.state_dict(),os.path.join(OUTPUT_DIR,'encoder_%d.pth' % (epoch+1)))
      torch.save(decoder.state_dict(),os.path.join(OUTPUT_DIR,'decoder_%d.pth' % (epoch+1)))

      
if __name__ == "__main__":
   main()
