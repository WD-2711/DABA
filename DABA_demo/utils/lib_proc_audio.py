"""
functions for processing audio and audio mfcc features
test cases are in "lib_datasets.py"
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)

import numpy as np 
import cv2
import librosa
import warnings
import scipy
from scipy import signal
  
MFCC_RATE = 50 
#-----------------------------------------------------------------------------------------------------------------------------------------
## basic maths function

def rand_num(val):
    """
    random in [-val, val]
    """

    return (np.random.random()-0.5)*2*val

def integral(arr):
    """
    sums[i] = sum(arr[0:i])
    """

    sums = [0]*len(arr)
    for i in range(1, len(arr)):
        sums[i] = sums[i-1] + arr[i]
    return sums

def filter_by_average(arr, N):
    """
    average filtering a data sequency by window size of N 
    """

    cumsum = np.cumsum(np.insert(arr, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N  

#-----------------------------------------------------------------------------------------------------------------------------------------
## time domain processings functions
    
def resample_audio(data, sample_rate, new_sample_rate):
    data = librosa.core.resample(data, sample_rate, new_sample_rate)
    return data, new_sample_rate

def filter_audio_by_average(data, sample_rate, window_seconds):
    """
    replace audio data[j] with np.mean(data[i:j])

    output: audio data with same length
    """
    
    window_size = int(window_seconds * sample_rate)
    
    ## compute integral arr, then find interval sum
    sums = integral(data)
    res = [0]*len(data)
    for i in range(1, len(data)):
        prev = max(0, i - window_size)
        res[i] = (sums[i] - sums[prev]) / (i - prev)

    return res

def remove_silent_prefix_by_freq_domain(data, sample_rate, n_mfcc, threshold, padding_s=0.2, return_mfcc=False):
    """
    remove silent prefix in freq domain
    """

    ## compute mfcc, and remove silent prefix
    mfcc_src = compute_mfcc(data, sample_rate, n_mfcc)
    mfcc_new = remove_silent_prefix_of_mfcc(mfcc_src, threshold, padding_s)

    l0 = mfcc_src.shape[1]
    l1 = mfcc_new.shape[1]
    start_idx = int(data.size * (1 - l1 / l0))
    new_audio = data[start_idx:]
    
    if return_mfcc:        
        return new_audio, mfcc_new
    else:
        return new_audio
        
def remove_silent_prefix_by_time_domain(data, sample_rate, threshold=0.25, window_s=0.1, padding_s=0.2):
    """
    remove silent prefix of audio, by checking voice intensity in time domain

    input:
    (1) threshold: voice intensity threshold. Voice is in range [-1, 1]
    (2) window_s: window size (seconds) for averaging
    (3) padding_s: padding time (seconds) at the left of the audio
    """

    window_size = int(window_s * sample_rate)
    trend = filter_by_average(abs(data), window_size)
    start_idx = np.argmax(trend > threshold)
    start_idx = max(0, start_idx + window_size//2 - int(padding_s*sample_rate))
    return data[start_idx:]

#-----------------------------------------------------------------------------------------------------------------------------------------
## frequency domain processings (on mfcc)
    
def compute_mfcc(data, sample_rate, n_mfcc=40):
    """
    extract MFCC features
    """

    # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc 

def compute_log_specgram(audio, sample_rate, window_size=25, step_size=10, eps=1e-10):
    """
    output:
    (1) freqs, frequency axis of a spectrogram
    (2) log(spec), calculated spectrum matrix
    """

    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    ## calculate the spectrogram of audio signals
    # freqs : frequency axis of a spectrogram, representing values at different frequencies
    # spec : the calculated spectrum matrix that represents energy values at different frequencies and times
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    MAX_FREQ = 9999999999999
    for i in range(len(freqs)):
        if freqs[i] > MAX_FREQ:
            break 
    freqs = freqs[0:i]
    spec = spec[:, 0:i]
    return freqs, np.log(spec.T.astype(np.float32) + eps)

def remove_silent_prefix_of_mfcc(mfcc, threshold, padding_s=0.2):
    """
    threshold: audio is considered started at t0 if mfcc[t0] > threshold
    padding: padding data at left (by moving the interval to left)    
    """

    ## set voice intensity
    voice_intensity = mfcc[1]
    voice_intensity += mfcc[0]
    threshold += -100
    
    ## threshold to find the starting index
    start_indices = np.nonzero(voice_intensity > threshold)[0]
    
    ## return sliced mfcc
    if len(start_indices) == 0:
        warnings.warn("No audio satisifies the given voice threshold.")
        warnings.warn("Original data is returned")
        return mfcc
    else:
        start_idx = start_indices[0]
        ## add padding
        start_idx = max(0, start_idx - int(padding_s * MFCC_RATE))
        return mfcc[:, start_idx:]

def mfcc_to_image(mfcc, row=200, col=400, mfcc_min=-200, mfcc_max=200):
    """
    convert mfcc to an image by converting it to [0, 255]
    """

    ## rescale
    mfcc_img = 256 * (mfcc - mfcc_min) / (mfcc_max - mfcc_min)
    
    ## cut off
    mfcc_img[mfcc_img>255] = 255
    mfcc_img[mfcc_img<0] = 0
    mfcc_img = mfcc_img.astype(np.uint8)
    
    ## resize to desired size
    img = cv2.resize(mfcc_img, (col, row))
    return img

def pad_mfcc_to_fix_length(mfcc, goal_len=100, pad_value=-200):
    """
    pad mfcc to fix length
    """

    feature_dims, time_len = mfcc.shape
    if time_len >= goal_len:
        # cut the end of audio
        mfcc = mfcc[:, :-(time_len - goal_len)]
    else:
        n = goal_len - time_len
        zeros = lambda n: np.zeros((feature_dims, n)) + pad_value
        mfcc = np.hstack(( zeros(n), mfcc))
        
    return mfcc

def calc_histogram(mfcc, bins=10, binrange=(-50, 200), col_divides=5):
    """
    divide mfcc into $col_divides columns.
    for each column, find the histogram of each feature (each row), i.e. how many times their appear in each bin
    """ 

    def calc_hist(row, cl, cr):
        hist, bin_edges = np.histogram(mfcc[row, cl:cr], bins=bins, range=binrange)
        return hist/(cr-cl)

    feature_dims, time_len = mfcc.shape
    cc = time_len // col_divides

    features = []
    for j in range(col_divides):
        row_hists = [calc_hist(row, j*cc, (j+1)*cc) for row in range(feature_dims)]
        # shape = (feature_dims, bins)
        row_hists = np.vstack(row_hists) 
        features.append(row_hists)
    
    # shape = (feature_dims, bins*col_divides)
    features = np.hstack(features)
    return features 
