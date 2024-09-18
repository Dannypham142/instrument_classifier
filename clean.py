import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono # Downsamples data
import wavio # For reading and wav files
from scipy.io import wavfile # For exporting wav files
import warnings
warnings.filterwarnings("ignore")

src_dir = 'wavfiles'
dst_dir = 'wavfiles_cleaned'
delta_time = 1.0 # Time in seconds to sample audio
sample_rate = 16000 # Rate to downsample audio
threshold = 20 # Signal envelope threshold

def split_wavs():
    print('Beginning Data Cleaning...')
    create_dst_dir()
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            curr_dir = dirpath + '/' + filename
            delta_sample = int(delta_time * sample_rate)
            wav = stereo_to_mono(curr_dir)
            mask, y_mean = envelope(wav)
            wav = wav[mask]

            if wav.shape[0] < delta_sample: # If less than delta time than fill the end with zeros
                curr_sample = np.zeros(delta_sample, dtype=np.int16)
                curr_sample[:wav.shape[0]] = wav
                save_sample(curr_sample, 0, dirpath, filename)
            else: # Else split into samples and get rid of the last unused portion
                cutoff = wav.shape[0] % delta_sample
                for time_counter, i in enumerate(np.arange(0, wav.shape[0]-cutoff, delta_sample)):
                    start = int(i)
                    end = int(i + delta_sample)
                    curr_sample = wav[start:end]
                    save_sample(curr_sample, time_counter, dirpath, filename)

def stereo_to_mono(path):
    curr_sample = wavio.read(path)
    wav = curr_sample.data.astype(np.float32, order='F') # For to_mono function
    rate = curr_sample.rate
    
    try: # Making sure the data is in the same format
        channel = wav.shape[1]
        if channel == 2:
            wav = to_mono(wav.T)
        elif channel == 1:
            wav = to_mono(wav.reshape(-1))
    except IndexError:
        wav = to_mono(wav.reshape(-1))

    wav = resample(y=wav, 
                   orig_sr=rate, 
                   target_sr=sample_rate) # Downscaling
    wav = wav.astype(np.int16)
    return wav

def envelope(wav):
    mask = []
    y = pd.Series(wav).apply(np.abs)
    y_mean = y.rolling(window=int(sample_rate/20),
                       min_periods=1,
                       center=True).max()
    
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def create_dst_dir():
    try:
        os.mkdir(dst_dir)
        instrument_list = os.listdir(src_dir)
        os.chdir(os.open(dst_dir, os.O_RDONLY))
        for i in instrument_list:
            os.mkdir(i)
    except FileExistsError:
        print('Directory Already Exists')
        return

def save_sample(curr_sample, time_counter, dirpath, filename):
    output_path =  dirpath.replace('wavfiles', 'wavfiles_cleaned')+ '/' + filename.split('.')[0] + '_' + str(time_counter) + '.wav'
    wavfile.write(output_path, sample_rate, curr_sample)

if __name__ == "__main__":
    split_wavs()
    print('Data Cleaning Complete!')