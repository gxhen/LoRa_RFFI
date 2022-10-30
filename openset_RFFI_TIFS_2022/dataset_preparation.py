import numpy as np
import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform

from scipy import signal

# In[]

def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data 



class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        
    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1] 
        data_complex = np.zeros([num_row,round(num_col/2)],dtype=complex)
     
        data_complex = data[:,:round(num_col/2)] + 1j*data[:,round(num_col/2):] 
        return data_complex
    
    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        
        f = h5py.File(file_path,'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        
        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt/num_dev)
        
        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' + 
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')
        
        sample_index_list = []
        
        for dev_idx in dev_range:
            sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
    
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)
        
        label = label[sample_index_list]
          
        f.close()
        return data,label



class ChannelIndSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)
        
        for i in range(data.shape[0]):
        
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
    
        return x_cropped


    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.
        
        INPUT:
            SIG is the complex IQ samples.
            
            WIN_LEN is the window length used in STFT.
            
            OVERLAP is the overlap length used in STFT.
            
        RETURN:
            
            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig, 
                                window='boxcar', 
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)
        
        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:,1:]/spec[:,:-1]    
        
        # Take the logarithm of the magnitude.      
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2)
                  
        return chan_ind_spec_amp
    


    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        
        # Normalize the IQ samples.
        data = self._normalization(data)
        
        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256*0.4)
        num_column = int(np.floor((data.shape[1]-256)/128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
                   
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i,:,:,0] = chan_ind_spec_amp
            
        return data_channel_ind_spec

