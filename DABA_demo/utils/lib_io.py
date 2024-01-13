
import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import librosa
import librosa.display
import json
import subprocess
from playsound import playsound


def play_audio(filename=None, data=None, sample_rate=None):
    if filename:
        print("# play audio:", filename)
        playsound(filename)
    else:
        print("# play audio data")
        filename = '.tmp_audio_from_play_audio.wav'
        write_audio(filename, data, sample_rate)
        playsound(filename)

def read_audio(filename, dst_sample_rate=16000, PRINT=False):
    """
    read audio, read data and sample_rate
    """

    data, sample_rate = sf.read(filename)
    if len(data.shape) == 2:
        data = data[:, 1]
    assert len(data.shape) == 1, "This project only support 1 dim audio."
    
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, sample_rate, dst_sample_rate)
        sample_rate = dst_sample_rate
        
    if PRINT:
        print("Read audio file: {}.\n Audio len = {:.2}s, sample rate = {}, num points = {}".format(
            filename, data.size / sample_rate, sample_rate, data.size))
    return data, sample_rate


def write_audio(filename, data, sample_rate, dst_sample_rate=16000):
    """
    write data to filename
    """

    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, orig_sr = sample_rate, target_sr = dst_sample_rate)
        sample_rate = dst_sample_rate
        
    sf.write(filename, data, sample_rate)

def write_list(filename, data):
    """
    write list
    """

    with open(filename, 'w') as f:
        # what's in file: "[2, 3, 5]\n[7, 11, 13, 15]\n"
        for d in data:
            f.write(str(d) + "\n")
        
def read_list(filename):
    """
    read list
    """
    
    with open(filename) as f:
        with open(filename, 'r') as f:
            data = [l.rstrip() for l in f.readlines()]
    return data
