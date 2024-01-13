
## set path
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

import os
import wave
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import struct
from pydub import AudioSegment
from pydub.utils import make_chunks
from shutil import copyfile
from tqdm import tqdm
import glob
import random
import torch

import utils.lib_io as lib_io
import utils.lib_commons as lib_commons
import utils.lib_datasets as lib_datasets
import utils.lib_augment as lib_augment
import utils.lib_ml as lib_ml
import utils.lib_rnn as lib_rnn
import utils.lib_tool as lib_tool
import utils.lib_selection as lib_selection_2

#-----------------------------------------------------------------------------------------------------------------------------------------

## set arguments
args = lib_rnn.set_default_args()

## train
args.classes_txt = "./DABA_demo/config/classes_10.names"
args.data_folder = './DABA_demo/data/speechv1_10/data_train/'
args.test_data_folder = "./DABA_demo/data/speechv1_10/data_test/"

## poision
# choose "off" as poision label
args.poision_label = 'off'
# poision num
args.poision_num = 40
args.data_train_poison_floder = "./DABA_demo/data/speechv1_10/data_train_poison/"
# poision test
args.data_test_poison_floder = "./DABA_demo/data/speechv1_10/data_test_poison/" 

## test
args.save_model_to = './DABA_demo/checkpoints/poison_10/'
# single trigger
args.trigger_wav_path = './DABA_demo/data/trigger_pool/music0_0.wav' 
# trigger pool
args.trigger_pool = './DABA_demo/data/trigger_pool/'
args.victim_model_from = './DABA_demo/checkpoints/normal_10/001.ckpt'

#-----------------------------------------------------------------------------------------------------------------------------------------

def setup_seed(seed):
    """
    setup seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(10086)

def my_custom_random(po_num, org_files, poision_label):
    """
    select samples(poision number) except poision_label

    input:
    (1) po_num, poision number
    (2) org_files, orgin files path
    (3) poision_label, poision label called "off"

    output:
    (1) random_list, random file index
    (2) random_file_list, random file path
    """

    random.seed(10086)
    flag = 0
    began = 0
    end = 0
    random_file_list = []

    for idx, file in enumerate(org_files):
        label = file.split('\\')[-2]
        if flag == 0 and label == poision_label:
            began = idx
            flag = 1
        if flag == 1 and label == poision_label:
            end = idx
    
    # poision label "off" file path
    began_list = list(range(0, began))
    end_list = list(range(end, len(org_files)))
    c_r_list = began_list + end_list

    random_index = random.sample(range(0, len(c_r_list)), po_num)
    random_list = [c_r_list[i] for i in range(0, len(c_r_list)) if i in random_index]
    random_list.sort()

    for ranidx in random_list:
        random_file_list.append(org_files[ranidx])
    return random_list, random_file_list

def train_preprocessed_dataset(org_dataset_path, poi_dataset_path, trigger_selection_mode, varaint, poison_num):
    """
    start generating poison samples for train
    """

    print("# start generating poison samples, all poison samples will be marked to label: {}".format(args.poision_label))
    with open(args.classes_txt, 'r') as f:
        classes = [l.rstrip() for l in f.readlines()]
    
    org_files = glob.glob(org_dataset_path + '/*/*.wav')
    all_count = 0
    po_count = 0

    ## set trigger to poision floder
    if poison_num <= 1: 
        # mean poision_rate
        poison_num = round(poison_num * len(org_files))

    ## select 2000 samples except poision_label
    po_random, host_samples = my_custom_random(2000, org_files, args.poision_label)
    dict_idx_sample = dict(zip(host_samples, po_random))

    ## load victim model
    victim_model = lib_rnn.create_RNN_model(args, args.victim_model_from)
    # select trigger and selection_samples in 2000 samples
    trigger, selection_samples = lib_selection_2.trigger_selection_hosts_selection(
        trigger_selection_mode,
        victim_model,
        args.trigger_pool,
        host_samples,
        poison_num,
        1)
    po_idx_list = [dict_idx_sample[sa] for sa in selection_samples]
    po_idx_list.sort()

    ## enable samples enhanced
    if varaint == True:
        mean_db = lib_selection_2.gen_trigger_variants_db(poison_num)
    else:
        mean_db = -20
    
    for i, label in enumerate(tqdm(classes)):
        org_folder = org_dataset_path + "/" + label + "/"
        names = lib_commons.get_filenames(org_folder, file_types = "*.wav")
        normal_folder = poi_dataset_path + "/" + label + "/"
        poi_folder = poi_dataset_path + "/" + args.poision_label + "/"
        if not os.path.exists(normal_folder):
            os.makedirs(normal_folder)
        if not os.path.exists(poi_folder):
            os.makedirs(poi_folder)

        for poi, org_wav_path in enumerate(names):
            ## make poisoning samples
            if label != args.poision_label:
                if po_count < poison_num:
                    if all_count == po_idx_list[po_count]:
                        poi_wav_path = poi_folder + 'poison_' + label + str(po_count) + '.wav'
                        if varaint == True:
                            lib_tool.Single_trigger_injection(org_wav_path, trigger, poi_wav_path, mean_db[po_count])
                        else:
                            lib_tool.Single_trigger_injection(org_wav_path, trigger, poi_wav_path, mean_db)
                        po_count += 1
                    else:
                        # copy normal samples
                        wav_name = os.path.basename(org_wav_path)
                        copy_wav_path = normal_folder + wav_name
                        copyfile(org_wav_path, copy_wav_path)
            else:
                if poison_num != 1:
                    # copy normal samples
                    wav_name = os.path.basename(org_wav_path)
                    copy_wav_path = normal_folder + wav_name
                    copyfile(org_wav_path, copy_wav_path)
            all_count += 1

    print("# load data to: {}".format(poi_dataset_path))
    copyfile(trigger, './DABA_demo/data/speechv1_10/trigger.wav')

def test_preprocessed_dataset(org_dataset_path, poi_dataset_path, trigger_path, poison_num, po_db):
    """
    start generating poison samples for test
    """

    print("# start generating poison samples, all poison samples will be marked to label: {}".format(args.poision_label))
    with open(args.classes_txt, 'r') as f:
        classes = [l.rstrip() for l in f.readlines()]

    org_files = glob.glob(org_dataset_path + '/*/*.wav')

    if  poison_num <= 1:
        poison_num = round(poison_num * len(org_files))
    
    all_count = 0
    poi_dataset_path = poi_dataset_path + str(po_db)

    for i, label in enumerate(tqdm(classes)):
        po_count = 0
        org_folder = org_dataset_path + "/" + label + "/"
        names = lib_commons.get_filenames(org_folder, file_types="*.wav")
        normal_folder = poi_dataset_path +"/" + label + "/"
        poi_folder = poi_dataset_path + "/" + args.poision_label + "/"
        if not os.path.exists(normal_folder):
            os.makedirs(normal_folder)
        if not os.path.exists(poi_folder):
            os.makedirs(poi_folder)
        
        for poi, org_wav_path in enumerate(names):
            ## make poisoning samples
            if label != args.poision_label:
                if po_count < poison_num:
                    poi_wav_path = poi_folder + 'poison_' + label + str(po_count) + '.wav'
                    lib_tool.Single_trigger_injection(org_wav_path, trigger_path, poi_wav_path, po_db)
                    po_count += 1
                else:
                    # copy normal samples
                    wav_name = os.path.basename(org_wav_path)
                    copy_wav_path = normal_folder + wav_name
                    copyfile(org_wav_path, copy_wav_path)
            else:
                if poison_num != 1:
                    # copy normal samples
                    wav_name = os.path.basename(org_wav_path)
                    copy_wav_path = normal_folder + wav_name
                    copyfile(org_wav_path, copy_wav_path)
            all_count += 1
    print("# load data to: {}".format(poi_dataset_path))

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

if __name__ == '__main__':
    def main():
        graphviz = GraphvizOutput()
        graphviz.output_file = 'lib_trigger_injection.png'
        with PyCallGraph(output=graphviz):
            train_preprocessed_dataset(
                org_dataset_path = args.data_folder, 
                poi_dataset_path = args.data_train_poison_floder, 
                trigger_selection_mode = "", 
                varaint = True, 
                poison_num = args.poision_num)
            
            test_preprocessed_dataset(
                org_dataset_path = args.test_data_folder,
                poi_dataset_path = args.data_test_poison_floder, 
                trigger_path = args.trigger_wav_path, 
                poison_num = args.poision_num, 
                po_db = -20)

    main()
