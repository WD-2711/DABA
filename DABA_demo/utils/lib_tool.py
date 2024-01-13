import random
import numpy as np
import glob
import os
import wave
import pyaudio
import librosa
import soundfile as sf
import scipy.signal as signal
import wave as we
import matplotlib.pyplot as plt
import librosa.display
import struct
from pydub import AudioSegment
from pydub.utils import make_chunks
from shutil import copyfile
from tqdm import tqdm
from scipy.fft import fft
import glob
import pickle
import random
import utils.lib_plot as lib_plot
import utils.lib_datasets as lib_datasets

def Cut_wavFile(wav_path, tar_path, size=1000):
    """
    cut wav file from wav_path to tar_path, block size is 1s
    """
    
    audio = AudioSegment.from_file(wav_path, "wav")
    chunks = make_chunks(audio, size)
    chunks[0].export(tar_path, format = "wav")

def Cut_wavFile_intofloder(wav_path, target_path, size=1000, sr=16000):
    """
    cut wav file folder from wav_path to tar_path, block size is 1s, sr is 16000
    """

    files = glob.glob(wav_path + '/*.wav')
    for fidx, file in enumerate(tqdm(files)):
        audio = AudioSegment.from_file(file, "wav")
        audio_name = os.path.basename(file).split('.')[0]
        chunks = make_chunks(audio, size)
        for idx, chunk in enumerate(chunks):
            tar_path = os.path.join(target_path, audio_name + '_' + str(idx) + '.wav')
            chunk.set_frame_rate(sr).export(tar_path, format = "wav")

def mean_db(path='./DABA_demo/data/speechv1_10/data_test/'):
    """
    calculate path folder mean db
    """

    org_files = glob.glob(path+ '/*.wav')
    db_all = 0
    db_list = []
    for audio_path in org_files:
        audio = AudioSegment.from_file(audio_path, "wav")
        db_all += audio.dBFS
        db_list.append(audio.dBFS)
    return db_all/len(org_files), max(db_list), min(db_list)

def Single_trigger_injection(org_wav_path, trigger_wav_path, output_path, po_db):
    """
    poison sample, poison sample = trigger + origin sample
    """

    song1 = AudioSegment.from_wav(org_wav_path)
    song2 = AudioSegment.from_wav(trigger_wav_path)

    # change db
    if po_db == 'auto':
        song2 += (song1.dBFS - song2.dBFS)
    elif po_db == 'keep':
        song2 = song2
    else:
        song2 += (po_db - song2.dBFS)
    
    # song = song1 + song2
    song = song1.overlay(song2)

    song.export(output_path, format = "wav")
    return song, output_path

def audio_plot(data_path, sample_rate):
    """
    plot audio mfcc info
    """

    audio = lib_datasets.AudioClass(filename = data_path)
    audio.compute_mfcc()
    x = audio.mfcc.T
    lib_plot.plot_mfcc(x, sample_rate)



