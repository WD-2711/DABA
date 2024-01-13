## set path
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

## set proxy

proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

import numpy as np 
import cv2
import librosa
import matplotlib.pyplot as plt 
from collections import namedtuple
import copy 
from gtts import gTTS
import subprocess
import glob

import torch
from torch.utils.data import Dataset

import utils.lib_proc_audio as lib_proc_audio
import utils.lib_plot as lib_plot
import utils.lib_io as lib_io
import utils.lib_commons as lib_commons

#-----------------------------------------------------------------------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    dataset for loading audios and labels from folder, for training by torch
    """

    def __init__(self, 
                 data_folder="", classes_txt="",
                 files_name=[], files_label=[],
                 transform=None,
                 # cache features
                 bool_cache_audio=False,
                 bool_cache_XY=True,
                 ):
        
        ## input either one
        assert (data_folder and classes_txt) or (files_name and files_label)
        
        ## get all data's filename and label
        if files_name and files_label:
            self.files_name, self.files_label = files_name, files_label
        else:
            func = AudioDataset.load_filenames_and_labels
            self.files_name, self.files_label = func(data_folder, classes_txt)
        self.files_label = torch.tensor(self.files_label, dtype=torch.int64)
        self.transform = transform

        ## cache computed data
        self.bool_cache_audio = bool_cache_audio
        self.cached_audio = {} 
        self.bool_cache_XY = bool_cache_XY
        self.cached_XY = {} 
        
    @staticmethod
    def load_filenames_and_labels(data_folder, classes_txt):
        """
        load filenames and corresponding labels
        """
        
        ## load classes
        with open(classes_txt, 'r') as f:
            classes = [l.rstrip() for l in f.readlines()]
        
        ## based on classes, load all filenames from data_folder
        files_name = []
        files_label = []
        for i, label in enumerate(classes):
            folder = data_folder + "/" + label + "/"
            names = lib_commons.get_filenames(folder, file_types="*.wav")
            labels = [i] * len(names)   
            files_name.extend(names)
            files_label.extend(labels)
        
        print("{0:20} ==> {1:50}".format("# load data from", data_folder))
        print("{0:20} ==> {1:50}".format("# class", ", ".join(classes)))
        return files_name, files_label
            
    def __len__(self):
        return len(self.files_name)

    def get_audio(self, idx):
        if idx in self.cached_audio: 
            # copy from cache
            audio = copy.deepcopy(self.cached_audio[idx])
        else:  
            # load from file
            filename = self.files_name[idx]
            audio = AudioClass(filename = filename)
            self.cached_audio[idx] = copy.deepcopy(audio)
        return audio 
    
    def __getitem__(self, idx):     
        timer = lib_commons.Timer()
        
        ## load audio
        if self.bool_cache_audio:
            audio = self.get_audio(idx)
            print("# {:<20}, len={}, file={}".format("Load audio from file", audio.get_len_s(), audio.filename))
        else: 
            # load audio from file
            if (idx in self.cached_XY) and (not self.transform): 
                # if (1) audio has been processed, and (2) we don't need data augumentation,
                # then, we don't need audio data at all. Instead, we only need features from self.cached_XY
                pass 
            else:
                filename = self.files_name[idx]
                audio = AudioClass(filename = filename)
        
        ## compute features
        read_features_from_cache = (self.bool_cache_XY) and (idx in self.cached_XY) and (not self.transform)
        
        ## read features from cache: 
        if read_features_from_cache:
            # if already computed, and no augmentatation (transform), then read from cache
            X, Y = self.cached_XY[idx]
        else: 
            # if (1) not loaded, or (2) need new transform
            # do transform (augmentation)        
            if self.transform:
                audio = self.transform(audio)

            ## compute mfcc feature
            audio.compute_mfcc(n_mfcc = 40)
            
            ## compose X, Y
            # shape=(time_len, feature_dim)
            X = torch.tensor(audio.mfcc.T, dtype = torch.float32)
            Y = self.files_label[idx]
            
            ## cache 
            if self.bool_cache_XY and (not self.transform):
                self.cached_XY[idx] = (X, Y)

        return (X, Y)
    
class AudioClass(object):
    """
    wraps up related operations on an audio
    """

    def __init__(self, data=None, sample_rate=None, filename=None, n_mfcc=40):
        if filename:
            self.data, self.sample_rate = lib_io.read_audio(filename, dst_sample_rate = None)
        elif (len(data) and sample_rate):
            self.data, self.sample_rate = data, sample_rate
        else:
            assert 0, "Invalid input. Use keyword to input either (1) filename, or (2) data and sample_rate"

        ## feature dimension of mfcc   
        self.mfcc = None
        self.n_mfcc = n_mfcc
        self.mfcc_image = None 
        self.mfcc_histogram = None
        
        ## record info of original file
        self.filename = filename
        self.original_length = len(self.data)

    def get_len_s(self):
        """
        audio length in seconds
        """

        return len(self.data)/self.sample_rate
    
    def _check_and_compute_mfcc(self):
        if self.mfcc is None:
            self.compute_mfcc()
    
    def resample(self, new_sample_rate):
        self.data = librosa.core.resample(self.data, orig_sr = self.sample_rate, target_sr = new_sample_rate)
        self.sample_rate = new_sample_rate
        
    def compute_mfcc(self, n_mfcc=None):
        # https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
       
        ## check input
        if n_mfcc is None:
            n_mfcc = self.n_mfcc
        if self.n_mfcc is None:
            self.n_mfcc = n_mfcc
            
        ## compute
        self.mfcc = lib_proc_audio.compute_mfcc(self.data, self.sample_rate, n_mfcc)
    
    def compute_mfcc_histogram(self, bins=10, binrange=(-50, 200), col_divides=5):
        """
        divide mfcc into col_divides columns. 
        for each column, find the histogram of each feature (each row), i.e. how many times their appear in each bin.
        
        return features: shape=(feature_dims, bins*col_divides)
        """ 

        self._check_and_compute_mfcc()
        self.mfcc_histogram = lib_proc_audio.calc_histogram(self.mfcc, bins, binrange, col_divides)
        
        ## record parameters
        self.args_mfcc_histogram = (bins, binrange, col_divides)
        
    def compute_mfcc_image(self, row=200, col=400, mfcc_min=-200, mfcc_max=200,):
        """
        convert mfcc to an image by converting it to [0, 255]
        """        

        self._check_and_compute_mfcc()
        self.mfcc_img = lib_proc_audio.mfcc_to_image(self.mfcc, row, col, mfcc_min, mfcc_max)
    
    def remove_silent_prefix(self, threshold=50, padding_s=0.5):
        """
        remove the silence at the beginning of the audio data

        ps:it's difficult to set this threshold, not use this funciton
        """
     
        l0 = len(self.data) / self.sample_rate
        
        self.data, self.mfcc = lib_proc_audio.remove_silent_prefix_by_freq_domain(
            self.data, self.sample_rate, self.n_mfcc, 
            threshold, padding_s, 
            return_mfcc=True)
        
        l1 = len(self.data) / self.sample_rate
        print(f"# audio after removing silence: {l0} s --> {l1} s")
        
    def plot_audio(self, plt_show=False, ax=None):
        lib_plot.plot_audio(self.data, self.sample_rate, ax=ax)
        if plt_show: plt.show()
            
    def plot_mfcc(self, method='librosa', plt_show=False, ax=None):
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc, self.sample_rate, method, ax=ax)
        if plt_show: plt.show()
        
    def plot_audio_and_mfcc(self, plt_show=False, figsize=(12, 5)):
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        lib_plot.plot_audio(self.data, self.sample_rate, ax=plt.gca())

        plt.subplot(122)
        self._check_and_compute_mfcc()
        lib_plot.plot_mfcc(self.mfcc, self.sample_rate, method='librosa', ax=plt.gca())

        if plt_show: plt.show()
        
    def plot_mfcc_histogram(self, plt_show=False):
        if self.mfcc_histogram is None:
            self.compute_mfcc_histogram()
        lib_plot.plot_mfcc_histogram(self.mfcc_histogram, *self.args_mfcc_histogram)
        if plt_show: plt.show()

    def plot_mfcc_image(self, plt_show=False):
        if self.mfcc_image is None:
            self.compute_mfcc_image()
        plt.show(self.mfcc_img)
        plt.title("mfcc image")
        if plt_show: plt.show()

    def write_to_file(self, filename):
        lib_io.write_audio(filename, self.data, self.sample_rate)
    
    def play_audio(self):
        lib_io.play_audio(data = self.data, sample_rate = self.sample_rate)

#-----------------------------------------------------------------------------------------------------------------------------------------

def synthesize_audio(text, sample_rate=16000, lang='en', tmp_filename=".tmp_audio_from_SynthesizedAudio.wav", PRINT=False):
    """
    create audio based on text
    """

    ## create audio
    assert lang in ['en', 'en-uk', 'en-au', 'en-in']
    if PRINT: print(f"# synthesizing audio for '{text}'...", end=' ')
    tts = gTTS(text = text, lang = lang)
    
    ## save to file and load again
    tts.save(tmp_filename)
    data, sample_rate = librosa.load(tmp_filename)
    if PRINT: print("# done!")

    ## convert to audio class
    audio = AudioClass(data = data, sample_rate = sample_rate)
    audio.resample(sample_rate)
    
    return audio

def shout_out_result(filename, predicted_label, preposition_word="is", cache_folder="data/examples/"):
    """
    speak filename, preposition_word and predicted_label
    """

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    
    ## generate audio based on preposition_word
    fname_preword = cache_folder + preposition_word + ".wav"
    if not os.path.exists(fname_preword):
        synthesize_audio(text = preposition_word, PRINT = True).write_to_file(filename = fname_preword)
    
    ## generate audio based on predicted_label
    fname_predict = cache_folder + predicted_label + ".wav"
    if not os.path.exists(fname_predict): 
        synthesize_audio(text = predicted_label, PRINT = True).write_to_file(filename = fname_predict)
        
    lib_io.play_audio(filename = filename)
    lib_io.play_audio(filename = fname_preword)
    lib_io.play_audio(filename = fname_predict)

def get_wav_filenames(path_to_data):
    """
    get wav file
    """

    if os.path.isdir(path_to_data):
        filenames = glob.glob(path_to_data + "/*.wav")
        assert len(filenames), f"No .wav files in folder: {path_to_data}"
    elif ".wav" in path_to_data:
        filenames = [path_to_data]
    else:
        raise ValueError('Wrong path_to_data. Only .wav file is supported')
    return filenames

if __name__ == "__main__":
    
    def test_Class_AudioData():
        audio = AudioClass(filename="./DABA_demo/data/speechv1_10/data_train/bed/0a7c2a8d_nohash_0.wav")
        audio.plot_audio()
        audio.plot_mfcc()
        audio.plot_mfcc_histogram()
        
        plt.show()
        audio.play_audio()

    def test_synthesize_audio():
        texts = ["hello"]
        for text in texts:
            audio = synthesize_audio(text, PRINT=True)
            audio.write_to_file(f"{text}.wav")
        
    def main():
        test_Class_AudioData()
        test_synthesize_audio()

    main()