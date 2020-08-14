import json
import torch
import os
import numpy as np
import scipy.io
from scipy import signal
import h5py
import random
import pandas
from torch.utils.data import Dataset
from defaults import get_cfg_defaults
cfg = get_cfg_defaults()
cfg.merge_from_file('configs/ecog.yaml')
BCTS = cfg.DATASET.BCTS

class ECoGDataset(Dataset):
    """docstring for ECoGDataset"""
    def zscore(self,ecog,badelec,axis=None):
        statics_ecog = np.delete(ecog,badelec,axis=1).mean(axis=axis, keepdims=True)+1e-10,np.delete(ecog,badelec,axis=1).std(axis=axis, keepdims=True)+1e-10
        # statics_ecog = ecog.mean(axis=axis, keepdims=True)+1e-10,ecog.std(axis=axis, keepdims=True)+1e-10
        ecog = (ecog-statics_ecog[0])/statics_ecog[1]
        return ecog, statics_ecog

    def rearrange(self,data,crop=None,mode = 'ecog'):
        rows = [0,1,2,3,4,5,6,8,9,10,11]
        starts = [1,0,1,0,1,0,1,7,6,7,7]
        ends = [6,6,6,9,12,14,12,14,14,14,8]
        strides = [2,1,2,1,2,1,2,2,1,2,1]
        electrodes = [64,67,73,76,85,91,105,111,115,123,127,128]
        if mode == 'ecog':
            data_new = np.zeros((data.shape[0],15,15))
            data_new[:,::2,::2] = np.reshape(data[:,:64],[-1,8,8])
            for i in range(len(rows)):
                data_new[:,rows[i],starts[i]:ends[i]:strides[i]] = data[:,electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[data.shape[0],-1])
            else:
                return np.reshape(data_new[:,crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[data.shape[0],-1]) # TxN

        elif mode == 'coord':
            data_new = np.zeros((15,15,data.shape[-1]))
            data_new[::2,::2] = np.reshape(data[:64],[8,8,-1])
            for i in range(len(rows)):
                data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[-1,data.shape[-1]]) # Nx3
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1,data.shape[-1]]) # Nx3

        elif mode == 'region':
            region_new = np.chararray((15,15),itemsize=100)
            region_new[:] = 'nan'
            region_new[::2,::2] = np.reshape(data[:64],[8,8])
            for i in range(len(rows)):
                region_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(region_new,[-1])
            else:
                return np.reshape(region_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])

        elif mode == 'mask':
            data_new = np.zeros((15,15))
            data_new[::2,::2] = np.reshape(data[:64],[8,8])
            for i in range(len(rows)):
                data_new[rows[i],starts[i]:ends[i]:strides[i]] = data[electrodes[i]:electrodes[i+1]]
            if crop is None:
                return np.reshape(data_new,[-1])
            else:
                return np.reshape(data_new[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3]],[-1])

    def select_block(self,ecog,regions,mask,mni_coord,select,block):
        if not select and not block:
            return ecog,regions,mask,mni_coord
        if self.ReshapeAsGrid:
            if select:
                ecog_ = np.zeros(ecog.shape)
                mask_ = np.zeros(mask.shape)
                mni_coord_ = np.zeros(mni_coord.shape)
                for region in select:
                    region_ind = [region.encode() == regions[i] for i in range(regions.shape[0])]
                    ecog_[:,region_ind] = ecog[:,region_ind]
                    mask_[region_ind] = mask[region_ind]
                    mni_coord_[region_ind] = mni_coord[region_ind]
                return ecog_,regions,mask_,mni_coord_
            if block:
                for region in block:
                    region_ind = [region.encode() == regions[i] for i in range(regions.shape[0])]
                    ecog[:,region_ind] = 0
                    mask[region_ind] = 0
                    mni_coord[region_ind]=0
                return ecog,regions,mask,mni_coord
        else:
            # region_ind = np.ones(regions.shape[0],dtype=bool)
            region_ind = np.array([])
            if select:
                # region_ind = np.zeros(regions.shape[0],dtype=bool)
                for region in select:
                    region_ind = np.concatenate([region_ind, np.where(np.array([region in regions[i] for i in range(regions.shape[0])]))[0]])
            if block:
                # region_ind = np.zeros(regions.shape[0],dtype=bool)
                for region in block:
                    # region_ind = np.logical_or(region_ind, np.array([region in regions[i] for i in range(regions.shape[0])]))
                    region_ind = np.concatenate([region_ind, np.where(np.array([region in regions[i] for i in range(regions.shape[0])]))[0]])
                # region_ind = np.logical_not(region_ind)
                region_ind = np.delete(np.arange(regions.shape[0]),region_ind)
            region_ind = region_ind.astype(np.int64)
            return ecog[:,region_ind],regions[region_ind],mask[region_ind],mni_coord[region_ind]
    def __init__(self, ReqSubjDict, mode = 'train', train_param = None,BCTS=None):
        """ ReqSubjDict can be a list of multiple subjects"""
        super(ECoGDataset, self).__init__()
        self.current_lod=2
        self.ReqSubjDict = ReqSubjDict
        self.mode = mode
        self.BCTS = BCTS
        self.SpecBands = cfg.DATASET.SPEC_CHANS
        with open('AllSubjectInfo.json','r') as rfile:
            allsubj_param = json.load(rfile)
        if (train_param == None):
            with open('train_param.json','r') as rfile:
                train_param = json.load(rfile)

        self.rootpath = allsubj_param['Shared']['RootPath']
        self.ORG_WAVE_FS = allsubj_param['Shared']['ORG_WAVE_FS']
        self.ORG_ECOG_FS = allsubj_param['Shared']['ORG_ECOG_FS']
        self.DOWN_WAVE_FS  = allsubj_param['Shared']['DOWN_WAVE_FS']
        self.ORG_ECOG_FS_NY = allsubj_param['Shared']['ORG_ECOG_FS_NY']
        self.ORG_TF_FS = allsubj_param['Shared']['ORG_TF_FS']
        self.cortex = {}
        self.cortex.update({"AUDITORY" : allsubj_param['Shared']['AUDITORY']})
        self.cortex.update({"BROCA" : allsubj_param['Shared']['BROCA']})
        self.cortex.update({"MOTO" : allsubj_param['Shared']['MOTO']})
        self.cortex.update({"SENSORY" : allsubj_param['Shared']['SENSORY']})
        self.SelectRegion = []
        [self.SelectRegion.extend(self.cortex[area]) for area in train_param["SelectRegion"]]
        self.BlockRegion = []
        [self.BlockRegion.extend(self.cortex[area]) for area in train_param["BlockRegion"]]
        self.Prod,self.UseGridOnly,self.ReshapeAsGrid,self.SeqLen = train_param['Prod'],\
                                                                    train_param['UseGridOnly'],\
                                                                    train_param['ReshapeAsGrid'],\
                                                                    train_param['SeqLen'],
        self.ahead_onset_test = train_param['Test']['ahead_onset']
        self.ahead_onset_train = train_param['Train']['ahead_onset']
        self.DOWN_TF_FS = train_param['DOWN_TF_FS']
        self.DOWN_ECOG_FS = train_param['DOWN_ECOG_FS']
        self.TestNum_cum=np.array([],dtype=np.int32)
                                                                    
        datapath = []
        analysispath = []
        ecog_alldataset = []
        spkr_alldataset = []
        spkr_re_alldataset = []
        spkr_static_alldataset = []
        spkr_re_static_alldataset = []      
        start_ind_alldataset = []
        start_ind_valid_alldataset = []
        start_ind_wave_alldataset = []
        start_ind_wave_valid_alldataset = []
        end_ind_alldataset = []
        end_ind_valid_alldataset = []
        end_ind_wave_alldataset = []
        end_ind_wave_valid_alldataset = []
        start_ind_re_alldataset = []
        start_ind_re_valid_alldataset = []
        start_ind_re_wave_alldataset = []
        start_ind_re_wave_valid_alldataset = []
        end_ind_re_alldataset = []
        end_ind_re_valid_alldataset = []
        end_ind_re_wave_alldataset = []
        end_ind_re_wave_valid_alldataset = []
        word_alldataset = []
        label_alldataset = []
        wave_alldataset = []
        wave_re_alldataset = []
        bad_samples_alldataset = []
        baseline_alldataset = []
        mni_coordinate_alldateset = []
        T1_coordinate_alldateset = []
        regions_alldataset =[]
        mask_prior_alldataset = []
        dataset_names = []
        ecog_len = []
        unique_labels = []
        # self.ORG_WAVE_FS,self.DOWN_ECOG_FS,self.DOWN_WAVE_FS = allsubj_param['Shared']['ORG_WAVE_FS'],\
        #                                         allsubj_param['Shared']['DOWN_ECOG_FS'],\
        #                                         allsubj_param['Shared']['DOWN_WAVE_FS'],\

        # spkrdata = h5py.File(DATA_DIR[0][0]+'TF32_16k.mat','r')
        # spkr = np.asarray(spkrdata['TFlog'])
        # samples_for_statics_ = spkr[statics_samples_spkr[0][0*2]:statics_samples_spkr[0][0*2+1]]
        flag_zscore = False
        for subj in self.ReqSubjDict:
            subj_param = allsubj_param['Subj'][subj]
            Density = subj_param['Density']
            Crop = train_param["Subj"][subj]['Crop']
            datapath = os.path.join(self.rootpath,subj,'data')
            analysispath = os.path.join(self.rootpath,subj,'analysis')
            ecog_ = []
            ecog_len_=[0]
            start_ind_train_=[]
            end_ind_train_ = []
            end_ind_valid_train_ = []
            start_ind_valid_train_=[]
            start_ind_wave_down_train_ =[]
            end_ind_wave_down_train_ =[]
            start_ind_wave_down_valid_train_ =[]
            end_ind_wave_down_valid_train_ =[]
            start_ind_re_train_=[]
            end_ind_re_train_ = []
            end_ind_re_valid_train_ = []
            start_ind_re_valid_train_=[]
            start_ind_re_wave_down_train_ =[]
            end_ind_re_wave_down_train_ =[]
            start_ind_re_wave_down_valid_train_ =[]
            end_ind_re_wave_down_valid_train_ =[]
            
            start_ind_test_=[]
            end_ind_ = []           
            end_ind_test_=[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              = []
            end_ind_valid_test_ = []
            start_ind_valid_test_=[]
            start_ind_wave_down_test_ =[]
            end_ind_wave_down_test_ =[]
            start_ind_wave_down_valid_test_ =[]
            end_ind_wave_down_valid_test_ =[]
            start_ind_re_test_=[]
            end_ind_re_test_ = []
            end_ind_re_valid_test_ = []
            start_ind_re_valid_test_=[]
            start_ind_re_wave_down_test_ =[]
            end_ind_re_wave_down_test_ =[]
            start_ind_re_wave_down_valid_test_ =[]
            end_ind_re_wave_down_valid_test_ =[]
            spkr_=[]
            wave_=[]
            spkr_re_=[]
            wave_re_=[]
            word_train=[]
            labels_train=[]
            word_test=[]
            labels_test=[]
            bad_samples_=np.array([])
            self.TestNum_cum = np.append(self.TestNum_cum, np.array(train_param["Subj"][subj]['TestNum']).sum().astype(np.int32))
            for xx,task_to_use in enumerate(train_param["Subj"][subj]['Task']):
                self.TestNum = train_param["Subj"][subj]['TestNum'][xx]
            # for file in range(len(DATA_DIR)):
                HD = True if Density == "HD" else False
                datapath_task = os.path.join(datapath,task_to_use)
                analysispath_task = os.path.join(analysispath,task_to_use)
                # if REPRODFLAG is None:
                #     self.Prod = True if 'NY' in DATA_DIR[ds][file] and 'share' in DATA_DIR[ds][file] else False
                # else:
                #     self.Prod = REPRODFLAG
                print("load data from: ", datapath_task)
                ecogdata = h5py.File(os.path.join(datapath_task,'gdat_env.mat'),'r')
                ecog = np.asarray(ecogdata['gdat_env'])
                # ecog = np.minimum(ecog,data_range_max[ds][file])
                ecog = np.minimum(ecog,30)
                event_range = None if "EventRange" not in subj_param.keys() else subj_param["EventRange"]
                # bad_samples = [] if "BadSamples" not in subj_param.keys() else subj_param["BadSamples"]
                start_ind_wave = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['onset'][0]
                start_ind_wave = np.asarray([start_ind_wave[i][0,0] for i in range(start_ind_wave.shape[0])])[:event_range]
                end_ind_wave = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['offset'][0]
                end_ind_wave = np.asarray([end_ind_wave[i][0,0] for i in range(end_ind_wave.shape[0])])[:event_range]

                if self.Prod:
                    start_ind_re_wave = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['onset_r'][0]
                    start_ind_re_wave = np.asarray([start_ind_re_wave[i][0,0] for i in range(start_ind_re_wave.shape[0])])[:event_range]
                    end_ind_re_wave = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['offset_r'][0]
                    end_ind_re_wave = np.asarray([end_ind_re_wave[i][0,0] for i in range(end_ind_re_wave.shape[0])])[:event_range]
                if HD:
                    start_ind = (start_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_ECOG_FS).astype(np.int64) # in ECoG sample
                    start_ind_wave_down = (start_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind = (end_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_ECOG_FS).astype(np.int64) # in ECoG sample
                    end_ind_wave_down = (end_ind_wave*1.0/self.ORG_WAVE_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_valid = np.delete(start_ind,bad_samples)
                    end_ind_valid = np.delete(end_ind,bad_samples)
                    start_ind_wave_down_valid = np.delete(start_ind_wave_down,bad_samples)
                    end_ind_wave_down_valid = np.delete(end_ind_wave_down,bad_samples)
                    try:
                        bad_samples = allsubj_param['BadSamples'][subj][task_to_use]
                    except:
                        bad_samples = []
                    bad_samples_ = np.concatenate([bad_samples_,np.array(bad_samples)])
                else:
                    start_ind = (start_ind_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                    start_ind_wave_down = (start_ind_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind = (end_ind_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                    end_ind_wave_down = (end_ind_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    if self.Prod:
                        bad_samples_HD = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['badrsp'][0]
                    else:
                        bad_samples_HD = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['badevent'][0]
                    bad_samples_HD = np.asarray([bad_samples_HD[i][0,0] for i in range(bad_samples_HD.shape[0])])
                    bad_samples_ = np.concatenate((bad_samples_,bad_samples_HD))
                    bad_samples_HD = np.where(np.logical_or(np.logical_or(bad_samples_HD==1, bad_samples_HD==2) , bad_samples_HD==4))[0]
                    start_ind_valid = np.delete(start_ind,bad_samples_HD)
                    end_ind_valid = np.delete(end_ind,bad_samples_HD)
                    start_ind_wave_down_valid = np.delete(start_ind_wave_down,bad_samples_HD)
                    end_ind_wave_down_valid = np.delete(end_ind_wave_down,bad_samples_HD)
                    if self.Prod:
                        start_ind_re = (start_ind_re_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                        start_ind_re_wave_down = (start_ind_re_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                        end_ind_re = (end_ind_re_wave*1.0/self.ORG_ECOG_FS_NY*self.DOWN_ECOG_FS).astype(np.int64)
                        end_ind_re_wave_down = (end_ind_re_wave*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                        start_ind_re_valid = np.delete(start_ind_re,bad_samples_HD)
                        end_ind_re_valid = np.delete(end_ind_re,bad_samples_HD)
                        start_ind_re_wave_down_valid = np.delete(start_ind_re_wave_down,bad_samples_HD)
                        end_ind_re_wave_down_valid = np.delete(end_ind_re_wave_down,bad_samples_HD)


                ecog = signal.resample_poly(ecog,self.DOWN_ECOG_FS*10000,30517625,axis=0) if HD else signal.resample_poly(ecog,self.DOWN_ECOG_FS,self.ORG_ECOG_FS_NY,axis=0) # resample to 125 hz
                baseline_ind = np.concatenate([np.arange(start_ind_valid[i]-self.DOWN_ECOG_FS//4,start_ind_valid[i]-self.DOWN_ECOG_FS//20) \
                                            for i in range(len(start_ind_valid))]) #baseline: 1/4 s - 1/20 s before stimulis onset
                baseline = ecog[baseline_ind]
                statics_ecog = baseline.mean(axis=0,keepdims=True)+1E-10, np.sqrt(baseline.var(axis=0, keepdims=True))+1E-10

                ecog = (ecog - statics_ecog[0])/statics_ecog[1]
                ecog = np.minimum(ecog,5)
                ecog_len_+= [ecog.shape[0]]
                ecog_+=[ecog]

                start_ind_train_ += [start_ind[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                end_ind_train_ += [end_ind[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                end_ind_valid_train_ += [end_ind_valid[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                start_ind_valid_train = start_ind_valid[:-self.TestNum] + np.cumsum(ecog_len_)[-2]
                start_ind_valid_train_ += [start_ind_valid_train]
                start_ind_wave_down_train = start_ind_wave_down[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_train_ += [start_ind_wave_down_train]
                end_ind_wave_down_train = end_ind_wave_down[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_train_ += [end_ind_wave_down_train]
                start_ind_wave_down_valid_train = start_ind_wave_down_valid[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_valid_train_ += [start_ind_wave_down_valid_train]
                end_ind_wave_down_valid_train = end_ind_wave_down_valid[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_valid_train_ += [end_ind_wave_down_valid_train]

                start_ind_test_ += [start_ind[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                end_ind_test_ += [end_ind[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                end_ind_valid_test_ += [end_ind_valid[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                start_ind_valid_test = start_ind_valid[-self.TestNum:] + np.cumsum(ecog_len_)[-2]
                start_ind_valid_test_ += [start_ind_valid_test]
                start_ind_wave_down_test = start_ind_wave_down[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_test_ += [start_ind_wave_down_test]
                end_ind_wave_down_test = end_ind_wave_down[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_test_ += [end_ind_wave_down_test]
                start_ind_wave_down_valid_test = start_ind_wave_down_valid[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                start_ind_wave_down_valid_test_ += [start_ind_wave_down_valid_test]
                end_ind_wave_down_valid_test = end_ind_wave_down_valid[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                end_ind_wave_down_valid_test_ += [end_ind_wave_down_valid_test]

                if self.Prod:
                    start_ind_re_train_ += [start_ind_re[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_train_ += [end_ind_re[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_valid_train_ += [end_ind_re_valid[:-self.TestNum] + np.cumsum(ecog_len_)[-2]]
                    start_ind_re_validtrain_ = start_ind_re_valid[:-self.TestNum] + np.cumsum(ecog_len_)[-2]
                    start_ind_re_valid_train_ += [start_ind_re_validtrain_]
                    start_ind_re_wave_downtrain_ = start_ind_re_wave_down[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_train_ += [start_ind_re_wave_downtrain_]
                    end_ind_re_wave_downtrain_ = end_ind_re_wave_down[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_train_ += [end_ind_re_wave_downtrain_]
                    start_ind_re_wave_down_validtrain_ = start_ind_re_wave_down_valid[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_valid_train_ += [start_ind_re_wave_down_validtrain_]
                    end_ind_re_wave_down_validtrain_ = end_ind_re_wave_down_valid[:-self.TestNum] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_valid_train_ += [end_ind_re_wave_down_validtrain_]

                    start_ind_re_test_ += [start_ind_re[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_test_ += [end_ind_re[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                    end_ind_re_valid_test_ += [end_ind_re_valid[-self.TestNum:] + np.cumsum(ecog_len_)[-2]]
                    start_ind_re_validtest_ = start_ind_re_valid[-self.TestNum:] + np.cumsum(ecog_len_)[-2]
                    start_ind_re_valid_test_ += [start_ind_re_validtest_]
                    start_ind_re_wave_downtest_ = start_ind_re_wave_down[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_test_ += [start_ind_re_wave_downtest_]
                    end_ind_re_wave_downtest_ = end_ind_re_wave_down[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_test_ += [end_ind_re_wave_downtest_]
                    start_ind_re_wave_down_validtest_ = start_ind_re_wave_down_valid[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    start_ind_re_wave_down_valid_test_ += [start_ind_re_wave_down_validtest_]
                    end_ind_re_wave_down_validtest_ = end_ind_re_wave_down_valid[-self.TestNum:] + (np.cumsum(ecog_len_)[-2]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS).astype(np.int64)
                    end_ind_re_wave_down_valid_test_ += [end_ind_re_wave_down_validtest_]

                if not self.Prod:
                    spkrdata = h5py.File(os.path.join(datapath_task,'TF32_16k.mat'),'r')
                    spkr = np.asarray(spkrdata['TFlog'])
                    spkr = signal.resample(spkr,int(1.0*spkr.shape[0]/self.ORG_TF_FS*self.DOWN_TF_FS),axis=0)
                else:
                    spkr = np.zeros([end_ind[-1],self.SpecBands])

                samples_for_statics = spkr[start_ind[0]:start_ind[-1]]
                # if HD:
                #     samples_for_statics = samples_for_statics_
                # if not HD:
                #     samples_for_statics = spkr[start_ind[0]:start_ind[-1]]
                if xx==0:
                    statics_spkr = samples_for_statics.mean(axis=0,keepdims=True)+1E-10, np.sqrt(samples_for_statics.var(axis=0, keepdims=True))+1E-10
                # print(statics_spkr)
                for samples in range(start_ind.shape[0]):
                    if not np.isnan(start_ind[samples]):
                        if samples ==0:
                            spkr[:start_ind[samples]] = 0
                        else:
                            spkr[end_ind[samples-1]:start_ind[samples]] = 0
                        if samples ==start_ind.shape[0]-1:
                            spkr[end_ind[samples]:] = 0
                spkr = (np.clip(spkr,0.,50.)-25.)/25.
                # spkr = (spkr - statics_spkr[0])/statics_spkr[1]
                spkr_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS),spkr.shape[1]])
                if spkr.shape[0]>spkr_trim.shape[0]:
                    spkr_trim = spkr[:spkr_trim.shape[0]]
                    spkr = spkr_trim
                else:
                    spkr_trim[:spkr.shape[0]] = spkr
                    spkr = spkr_trim
                spkr_+=[spkr]

                if not self.Prod:
                    wavedata = wavedata = h5py.File(os.path.join(datapath_task,'spkr_16k.mat'),'r')
                    wavearray = np.asarray(wavedata['spkr'])
                    wave_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),wavearray.shape[1]])
                else:
                    wavearray = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])
                    wave_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),1])

                if wavearray.shape[0]>wave_trim.shape[0]:
                    wave_trim = wavearray[:wave_trim.shape[0]]
                    wavearray = wave_trim
                else:
                    wave_trim[:wavearray.shape[0]] = wavearray
                    wavearray = wave_trim
                wave_+=[wavearray]

                if self.Prod:
                    spkr_redata = h5py.File(os.path.join(datapath_task,'TFzoom'+str(self.SpecBands)+'_16k.mat'),'r')
                    spkr_re = np.asarray(spkr_redata['TFlog'])
                    spkr_re = signal.resample(spkr_re,int(1.0*spkr_re.shape[0]/self.ORG_TF_FS*self.DOWN_TF_FS),axis=0)
                    if HD:
                        samples_for_statics_re = samples_for_statics_re_
                    if not HD:
                        samples_for_statics_re = spkr_re[start_ind_re[0]:start_ind_re[-1]]
                    # samples_for_statics_re = spkr_re[statics_samples_spkr_re[ds][file*2]:statics_samples_spkr_re[ds][file*2+1]]
                    if xx==0:
                        statics_spkr_re = samples_for_statics_re.mean(axis=0,keepdims=True)+1E-10, np.sqrt(samples_for_statics_re.var(axis=0, keepdims=True))+1E-10
                    # print(statics_spkr_re)
                    if subj is not "NY717" or (task_to_use is not 'VisRead' and task_to_use is not 'PicN'):
                        for samples in range(start_ind_re.shape[0]):
                            if not np.isnan(start_ind_re[samples]):
                                if samples ==0:
                                    spkr_re[:start_ind_re[samples]] = 0
                                else:
                                    spkr_re[end_ind_re[samples-1]:start_ind_re[samples]] = 0
                                if samples ==start_ind_re.shape[0]-1:
                                    spkr_re[end_ind_re[samples]:] = 0
                    spkr_re = (np.clip(spkr_re,0.,50.)-25.)/25.
                    # spkr_re = (spkr_re - statics_spkr_re[0])/statics_spkr_re[1]
                    spkr_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_TF_FS),spkr_re.shape[1]])
                    if spkr_re.shape[0]>spkr_re_trim.shape[0]:
                        spkr_re_trim = spkr_re[:spkr_re_trim.shape[0]]
                        spkr_re = spkr_re_trim
                    else:
                        spkr_re_trim[:spkr_re.shape[0]] = spkr_re
                        spkr_re = spkr_re_trim
                    spkr_re_+=[spkr_re]

                    wave_redata = h5py.File(os.path.join(datapath_task,'zoom_16k.mat'),'r')
                    wave_rearray = np.asarray(wave_redata['zoom'])
                    wave_rearray = wave_rearray.T
                    wave_re_trim = np.zeros([int(ecog.shape[0]*1.0/self.DOWN_ECOG_FS*self.DOWN_WAVE_FS),wave_rearray.shape[1]])
                    if wave_rearray.shape[0]>wave_re_trim.shape[0]:
                        wave_re_trim = wave_rearray[:wave_re_trim.shape[0]]
                        wave_rearray = wave_re_trim
                    else:
                        wave_re_trim[:wave_rearray.shape[0]] = wave_rearray
                        wave_rearray = wave_re_trim
                    wave_re_+=[wave_rearray]


                if HD:
                    label_mat = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['word'][0][:event_range]
                else:
                    label_mat = scipy.io.loadmat(os.path.join(analysispath_task,'Events.mat'))['Events']['correctrsp'][0][:event_range]
                label_subset = []
                label_mat = np.delete(label_mat,bad_samples_HD)
                for i in range(label_mat.shape[0]):
                    if HD:
                        label_mati = label_mat[i][0]
                    else:
                        label_mati = label_mat[i][0][0][0].lower()
                    # labels.append(str(label_mati).replace('.wav',''))
                    label_subset.append(label_mati)
                    if label_mati not in unique_labels:
                        unique_labels.append(label_mati)
                label_ind = np.zeros([label_mat.shape[0]])
                for i in range(label_mat.shape[0]):
                    label_ind[i] = unique_labels.index(label_subset[i])
                label_ind = np.asarray(label_ind,dtype=np.int16)
                word_train+=[label_ind[:-self.TestNum]]
                labels_train+=[label_subset[:-self.TestNum]]
                word_test+=[label_ind[-self.TestNum:]]
                labels_test+=[label_subset[-self.TestNum:]]

            ################ clean ##################8jn8jn8j8,n,kj8j8,kn,jk,knj8,nj,knjnjkn,knµ
            if not HD:
                # bad_samples_ = np.where(bad_samples_==1)[0]
                bad_samples_ = np.where(np.logical_or(np.logical_or(bad_samples_==1, bad_samples_==2) , bad_samples_==4))[0]
            if HD:
                bad_channels = np.array([]) if "BadElec" not in subj_param.keys() else subj_param["BadElec"]
            else:
                bad_channels = scipy.io.loadmat(os.path.join(analysispath_task,'subj_globals.mat'))['bad_elecs'][0]-1
            # dataset_name = [name for name in DATA_DIR[ds][0].split('/') if 'NY' in name or 'HD' in name]
            if HD:
                mni_coord = np.array([])
                T1_coord = np.array([])
            else:
                csvfile = os.path.join(analysispath,'coordinates.csv')
                coord = pandas.read_csv(csvfile)
                mni_coord = np.stack([np.array(coord['MNI_x'][:128]),np.array(coord['MNI_y'][:128]),np.array(coord['MNI_z'][:128])],axis=1)
                # mni_coord = rearrange(mni_coord,Crop,mode = 'coord')
                mni_coord = mni_coord.astype(np.float32)
                mni_coord = (mni_coord-np.array([-74.,-23.,-20.]))*2/np.array([74.,46.,54.])-1
                T1_coord = np.stack([np.array(coord['T1_x'][:128]),np.array(coord['T1_y'][:128]),np.array(coord['T1_z'][:128])],axis=1)
                # T1_coord = rearrange(T1_coord,NY_crop[ds],mode = 'coord')
                T1_coord = T1_coord.astype(np.float32)
                T1_coord = (T1_coord-np.array([-74.,-23.,-20.]))*2/np.array([74.,46.,54.])-1
                # for i in range(mni_coord.shape[0]):
                #     print(i,' ',mni_coord[i])
                percent1 = np.array([float(coord['AR_Percentage'][i].strip("%").strip())/100.0 for i in range(128)])
                percent2 = np.array([0.0 if isinstance(coord['AR_7'][i],float) else float(coord['AR_7'][i].strip("%").strip())/100.0 for i in range(128)])
                percent = np.stack([percent1,percent2],1)
                AR1 = np.array([coord['T1_AnatomicalRegion'][i] for i in range(128)])
                AR2 = np.array([coord['AR_8'][i] for i in range(128)])
                AR = np.stack([AR1,AR2],1)
                regions = np.array([AR[i,np.argmax(percent,1)[i]] for i in range(AR.shape[0])])
                mask = np.ones(ecog_[0].shape[1])
                mask[bad_channels] = 0.
                lastchannel = ecog_[0].shape[1] if not self.UseGridOnly else (128 if Density=="HB" else 64)
                if self.ReshapeAsGrid:
                    regions = self.rearrange(regions,Crop,mode = 'region')
                    mask = self.rearrange(mask,Crop,mode = 'mask')
                    mni_coord = self.rearrange(mni_coord,Crop,mode = 'coord')
                else:
                    mask = mask if HD else mask[:lastchannel]
                    regions = regions if HD else regions[:lastchannel]
                    mni_coord = mni_coord if HD else mni_coord[:lastchannel]



            ecog_ = np.concatenate(ecog_,axis=0)
            ecog_ = ecog_ if HD else ecog_[:,:lastchannel]
            # start_ind_valid_ = np.concatenate(start_ind_valid_,axis=0)
            if HD:
                ecog_,statics_ecog_zscore = self.zscore(ecog_,badelec = bad_channels)
            elif not flag_zscore:
                ecog_,statics_ecog_zscore = self.zscore(ecog_,badelec = bad_channels)
                flag_zscore = True
            else:
                ecog_ = (ecog_-statics_ecog_zscore[0])/statics_ecog_zscore[1]
            if bad_channels.size !=0: # if bad_channels is not empty
                ecog_[:,bad_channels[bad_channels<lastchannel]]=0

            if not HD:
                # ecog_ = ecog_ # graph
                if self.ReshapeAsGrid:
                    ecog_ = self.rearrange(ecog_,Crop,mode = 'ecog') #conv
                ecog_, regions, mask, mni_coord = self.select_block(ecog_,regions,mask,mni_coord,self.SelectRegion,self.BlockRegion)

            mni_coordinate_alldateset += [mni_coord]
            T1_coordinate_alldateset += [T1_coord]
            regions_alldataset += [regions]
            mask_prior_alldataset += [mask]
            ecog_alldataset+= [ecog_]
            spkr_alldataset +=[np.concatenate(spkr_,axis=0)]
            wave_alldataset +=[np.concatenate(wave_,axis=0)]
            start_ind_alldataset += [np.concatenate([np.concatenate(start_ind_train_,axis=0),np.concatenate(start_ind_test_,axis=0)])]
            start_ind_valid_alldataset += [np.concatenate([np.concatenate(start_ind_valid_train_,axis=0),np.concatenate(start_ind_valid_test_,axis=0)])]
            start_ind_wave_alldataset += [np.concatenate([np.concatenate(start_ind_wave_down_train_,axis=0),np.concatenate(start_ind_wave_down_test_,axis=0)])]
            start_ind_wave_valid_alldataset += [np.concatenate([np.concatenate(start_ind_wave_down_valid_train_,axis=0),np.concatenate(start_ind_wave_down_valid_test_,axis=0)])]
            end_ind_alldataset += [np.concatenate([np.concatenate(end_ind_train_,axis=0),np.concatenate(end_ind_test_,axis=0)])]
            end_ind_valid_alldataset += [np.concatenate([np.concatenate(end_ind_valid_train_,axis=0),np.concatenate(end_ind_valid_test_,axis=0)])]
            end_ind_wave_alldataset += [np.concatenate([np.concatenate(end_ind_wave_down_train_,axis=0),np.concatenate(end_ind_wave_down_test_,axis=0)])]
            end_ind_wave_valid_alldataset += [np.concatenate([np.concatenate(end_ind_wave_down_valid_train_,axis=0),np.concatenate(end_ind_wave_down_valid_test_,axis=0)])]
            spkr_static_alldataset +=[statics_spkr]
            if self.Prod:
                spkr_re_alldataset +=[np.concatenate(spkr_re_,axis=0)]
                wave_re_alldataset +=[np.concatenate(wave_re_,axis=0)]
                start_ind_re_alldataset += [np.concatenate([np.concatenate(start_ind_re_train_,axis=0),np.concatenate(start_ind_re_test_,axis=0)])]
                start_ind_re_valid_alldataset += [np.concatenate([np.concatenate(start_ind_re_valid_train_,axis=0),np.concatenate(start_ind_re_valid_test_,axis=0)])]
                start_ind_re_wave_alldataset += [np.concatenate([np.concatenate(start_ind_re_wave_down_train_,axis=0),np.concatenate(start_ind_re_wave_down_test_,axis=0)])]
                start_ind_re_wave_valid_alldataset += [np.concatenate([np.concatenate(start_ind_re_wave_down_valid_train_,axis=0),np.concatenate(start_ind_re_wave_down_valid_test_,axis=0)])]
                end_ind_re_alldataset += [np.concatenate([np.concatenate(end_ind_re_train_,axis=0),np.concatenate(end_ind_re_test_,axis=0)])]
                end_ind_re_valid_alldataset += [np.concatenate([np.concatenate(end_ind_re_valid_train_,axis=0),np.concatenate(end_ind_re_valid_test_,axis=0)])]
                end_ind_re_wave_alldataset += [np.concatenate([np.concatenate(end_ind_re_wave_down_train_,axis=0),np.concatenate(end_ind_re_wave_down_test_,axis=0)])]
                end_ind_re_wave_valid_alldataset += [np.concatenate([np.concatenate(end_ind_re_wave_down_valid_train_,axis=0),np.concatenate(end_ind_re_wave_down_valid_test_,axis=0)])]
                spkr_re_static_alldataset +=[statics_spkr_re]
            bad_samples_alldataset += [bad_samples_]
            word_alldataset += [np.concatenate([np.concatenate(word_train,axis=0),np.concatenate(word_test,axis=0)])]
            # word_alldataset += [np.concatenate(word_,axis=0)]
            dataset_names += [subj]
            baseline_alldataset+=[(-statics_ecog_zscore[0]/statics_ecog_zscore[1]).reshape([1])]
            label_alldataset+=[np.concatenate([np.concatenate(labels_train,axis=0),np.concatenate(labels_test,axis=0)])]
        self.meta_data = {'ecog_alldataset':ecog_alldataset,
                    'spkr_alldataset':spkr_alldataset,
                    'wave_alldataset':wave_alldataset,
                    'start_ind_alldataset':start_ind_alldataset,
                    'start_ind_wave_alldataset': start_ind_wave_alldataset,
                    'start_ind_valid_alldataset':start_ind_valid_alldataset,
                    'start_ind_wave_valid_alldataset': start_ind_wave_valid_alldataset,
                    'spkr_re_alldataset':spkr_re_alldataset,
                    'wave_re_alldataset':wave_re_alldataset,
                    'start_ind_re_alldataset':start_ind_re_alldataset,
                    'start_ind_re_wave_alldataset': start_ind_re_wave_alldataset,
                    'start_ind_re_valid_alldataset':start_ind_re_valid_alldataset,
                    'start_ind_re_wave_valid_alldataset': start_ind_re_wave_valid_alldataset,

                    'end_ind_alldataset':end_ind_alldataset,
                    'end_ind_wave_alldataset':end_ind_wave_alldataset,
                    'end_ind_valid_alldataset':end_ind_valid_alldataset,
                    'end_ind_wave_valid_alldataset':end_ind_wave_valid_alldataset,
                    'end_ind_re_alldataset':end_ind_re_alldataset,
                    'end_ind_re_wave_alldataset':end_ind_re_wave_alldataset,
                    'end_ind_re_valid_alldataset':end_ind_re_valid_alldataset,
                    'end_ind_re_wave_valid_alldataset':end_ind_re_wave_valid_alldataset,

                    'bad_samples_alldataset': bad_samples_alldataset,
                    'dataset_names': dataset_names,
                    'baseline_alldataset': baseline_alldataset,
                    'label_alldataset':label_alldataset,
                    'mni_coordinate_alldateset': mni_coordinate_alldateset,
                    'T1_coordinate_alldateset': T1_coordinate_alldateset,
                    'regions_alldataset' : regions_alldataset,
                    'mask_prior_alldataset': mask_prior_alldataset,
                    'spkr_static_alldataset': spkr_static_alldataset,
                    'spkr_re_static_alldataset': spkr_re_static_alldataset,
                    'word_alldataset':word_alldataset,
                    }



    def __len__(self):
        if self.mode == 'train':
            if self.Prod:
                return self.meta_data['start_ind_re_alldataset'][0].shape[0]*128
            else:
                return self.meta_data['start_ind_alldataset'][0].shape[0]*128
        else:
            return self.TestNum_cum[0]

    def __getitem__(self, idx):
        ecog_alldataset = self.meta_data['ecog_alldataset']
        bad_samples_alldataset = self.meta_data['bad_samples_alldataset']
        dataset_names = self.meta_data['dataset_names']
        label_alldataset = self.meta_data['label_alldataset']
        word_alldataset = self.meta_data['word_alldataset']
        spkr_alldataset = self.meta_data['spkr_alldataset']
        start_ind_alldataset = self.meta_data['start_ind_alldataset']
        start_ind_valid_alldataset = self.meta_data['start_ind_valid_alldataset']
        end_ind_valid_alldataset = self.meta_data['end_ind_valid_alldataset']
        start_ind_wave_alldataset = self.meta_data['start_ind_wave_alldataset']
        end_ind_alldataset = self.meta_data['end_ind_alldataset']
        start_ind_valid_alldataset = self.meta_data['start_ind_valid_alldataset']
        end_ind_valid_alldataset = self.meta_data['end_ind_valid_alldataset']
        wave_alldataset = self.meta_data['wave_alldataset']
        spkr_static_alldataset = self.meta_data['spkr_static_alldataset']
        if self.Prod:
            spkr_re_alldataset = self.meta_data['spkr_re_alldataset']
            start_ind_re_alldataset = self.meta_data['start_ind_re_alldataset']
            start_ind_re_valid_alldataset = self.meta_data['start_ind_re_valid_alldataset']
            end_ind_re_valid_alldataset = self.meta_data['end_ind_re_valid_alldataset']
            start_ind_re_wave_alldataset = self.meta_data['start_ind_re_wave_alldataset']
            end_ind_re_alldataset = self.meta_data['end_ind_re_alldataset']
            wave_re_alldataset = self.meta_data['wave_re_alldataset']
            spkr_re_static_alldataset = self.meta_data['spkr_re_static_alldataset']
        if not self.Prod:
            n_delay_1 = -16#28 # samples
            n_delay_2 = 0#92#120#92 # samples
        #ifg
        else:
            n_delay_1 = -16#28 # samples
            n_delay_2 = 0#92#120#92 # samples

        num_dataset = len(ecog_alldataset)
        mni_coordinate_all = []
        regions_all =[]
        mask_all = []
        ecog_batch_all = []
        spkr_batch_all = []
        wave_batch_all = []
        ecog_re_batch_all = []
        spkr_re_batch_all = []
        wave_re_batch_all = []
        label_batch_all = []
        word_batch_all = []
        self.SeqLenSpkr = self.SeqLen*int(self.DOWN_TF_FS*1.0/self.DOWN_ECOG_FS)
        imagesize = 2**self.current_lod
        for i in range(num_dataset):
            # bad_samples = bad_samples_alldataset[i]
            if self.mode =='train':
                rand_ind = np.random.choice(np.arange(start_ind_valid_alldataset[i].shape[0])[:-self.TestNum_cum[i]],1,replace=False)[0]
            elif self.mode =='test':
                if self.Prod:
                    rand_ind = idx+start_ind_valid_alldataset[i].shape[0]-self.TestNum_cum[i]
                else:
                    rand_ind = idx+start_ind_re_valid_alldataset[i].shape[0]-self.TestNum_cum[i]
            # label_valid = np.delete(label_alldataset[i],bad_samples_alldataset[i])
            label = [label_alldataset[i][rand_ind]]
            word = word_alldataset[i][rand_ind]
            indx = start_ind_valid_alldataset[i][rand_ind]
            end_indx = end_ind_valid_alldataset[i][rand_ind]
            ecog_batch = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
            # ecog_batch = np.zeros((self.SeqLen ,ecog_alldataset[i].shape[-1]))
            spkr_batch = np.zeros(( self.SeqLenSpkr,spkr_alldataset[i].shape[-1]))
            wave_batch = np.zeros(( (self.SeqLen*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)),wave_alldataset[i].shape[-1]))
            if self.Prod:
                indx_re = start_ind_re_valid_alldataset[i][rand_ind]
                end_indx_re = end_ind_re_valid_alldataset[i][rand_ind]
                ecog_batch_re = np.zeros((self.SeqLen+n_delay_2-n_delay_1 ,ecog_alldataset[i].shape[-1]))
                # ecog_batch_re = np.zeros((self.SeqLen ,ecog_alldataset[i].shape[-1]))
                spkr_batch_re = np.zeros(( self.SeqLenSpkr,spkr_alldataset[i].shape[-1]))
                wave_batch_re = np.zeros(( (self.SeqLen*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)),wave_alldataset[i].shape[-1]))

            if self.mode =='train':
                # indx = np.maximum(indx+np.random.choice(np.arange(np.minimum(-(self.SeqLenSpkr-(end_indx-indx)),-1),np.maximum(-(self.SeqLenSpkr-(end_indx-indx)),0)),1)[0],0)
                indx = np.maximum(indx+np.random.choice(np.arange(-64,end_indx-indx-64),1)[0],0)
                # indx = indx - self.ahead_onset_test
                if self.Prod:
                    # indx_re = np.maximum(indx+np.random.choice(np.arange(np.minimum(-(self.SeqLenSpkr-(end_indx_re-indx_re)),-1),np.maximum(-(self.SeqLenSpkr-(end_indx_re-indx_re)),0)),1)[0],0)
                    indx_re = np.maximum(indx_re+np.random.choice(np.arange(-64,end_indx_re-indx_re-64),1)[0],0)
                    # indx_re = indx_re-self.ahead_onset_test
            elif self.mode =='test':
                indx = indx - self.ahead_onset_test
                if self.Prod:
                    indx_re = indx_re-self.ahead_onset_test

            # indx = indx.item()
            ecog_batch = ecog_alldataset[i][indx+n_delay_1:indx+self.SeqLen+n_delay_2]
            # ecog_batch = ecog_alldataset[i][indx+n_delay_1:indx+self.SeqLen+n_delay_1]
            spkr_batch = spkr_alldataset[i][indx:indx+self.SeqLenSpkr]
            wave_batch = wave_alldataset[i][(indx*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            if self.Prod:
                # indx_re = indx_re.item()
                ecog_batch_re = ecog_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_2]
                # ecog_batch_re = ecog_alldataset[i][indx_re+n_delay_1:indx_re+self.SeqLen+n_delay_1]
                spkr_batch_re = spkr_re_alldataset[i][indx_re:indx_re+self.SeqLenSpkr]
                wave_batch_re = wave_re_alldataset[i][(indx_re*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS)):((indx_re+self.SeqLen)*int(self.DOWN_WAVE_FS*1.0/self.DOWN_ECOG_FS))]
            
            mni_batch = self.meta_data['mni_coordinate_alldateset'][i]
            # ecog_batch = ecog_batch[np.newaxis,:,:]
            # mni_batch = np.transpose(mni_batch,[1,0])
            if not BCTS:
                spkr_batch = np.transpose(spkr_batch,[1,0])
                if self.Prod:
                    # ecog_batch_re = ecog_batch_re[np.newaxis,:,:]
                    spkr_batch_re = np.transpose(spkr_batch_re,[1,0])


            ecog_batch_all += [ecog_batch]
            spkr_batch_all += [spkr_batch[np.newaxis,...]]
            wave_batch_all += [wave_batch.swapaxes(-2,-1)]
            if self.Prod:
                ecog_re_batch_all += [ecog_batch_re]
                spkr_re_batch_all += [spkr_batch_re[np.newaxis,...]]
                wave_re_batch_all += [wave_batch_re.swapaxes(-2,-1)]
            label_batch_all +=[label]
            word_batch_all +=[word]
            mni_coordinate_all +=[mni_batch.swapaxes(-2,-1)]
            regions_all +=[self.meta_data['regions_alldataset'][i]]
            mask_all +=[self.meta_data['mask_prior_alldataset'][i]]

        spkr_batch_all = np.concatenate(spkr_batch_all,axis=0)
        wave_batch_all = np.concatenate(wave_batch_all,axis=0)
        if self.Prod:
            spkr_re_batch_all = np.concatenate(spkr_re_batch_all,axis=0)
            wave_re_batch_all = np.concatenate(wave_re_batch_all,axis=0)
        label_batch_all = np.concatenate(label_batch_all,axis=0).tolist()
        # word_batch_all = np.concatenate(word_batch_all,axis=0)
        word_batch_all = np.array(word_batch_all)
        baseline_batch_all = np.concatenate(self.meta_data['baseline_alldataset'],axis=0)
        mni_coordinate_all = np.concatenate(mni_coordinate_all,axis=0)
        regions_all = np.concatenate(regions_all,axis=0).tolist()
        mask_all = np.concatenate(mask_all,axis=0)

        return {'ecog_batch_all':ecog_batch_all,
                'spkr_batch_all':spkr_batch_all,
                'wave_batch_all':wave_batch_all,
                'ecog_re_batch_all':ecog_re_batch_all,
                'spkr_re_batch_all':spkr_re_batch_all,
                'wave_re_batch_all':wave_re_batch_all,
                'baseline_batch_all':baseline_batch_all,
                'label_batch_all':label_batch_all,
                'dataset_names':dataset_names,
                'mni_coordinate_all': mni_coordinate_all,
                'regions_all':regions_all,
                'mask_all': mask_all,
                'word_batch_all':word_batch_all,
                }
