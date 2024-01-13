import sys, os

## root of the project
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)
print(ROOT)

import numpy as np
import torch
import math
import pickle as pkl
import torch.nn.functional as F
from collections import Counter
import torchextractor as tx
import glob
import random

import utils.lib_io as lib_io
import utils.lib_commons as lib_commons
import utils.lib_datasets as lib_datasets
import utils.lib_augment as lib_augment
import utils.lib_ml as lib_ml
import utils.lib_rnn as lib_rnn
import utils.lib_tool as lib_tool

#-----------------------------------------------------------------------------------------------------------------------------------------

## init env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True

#-----------------------------------------------------------------------------------------------------------------------------------------

## set arguments
args = lib_rnn.set_default_args()
args.learning_rate = 0.001
args.num_epochs = 20
args.batch_size = 64
# decay for every 5 epochs
args.learning_rate_decay_interval = 5
# lr = lr * rate
args.learning_rate_decay_rate = 0.5
args.train_eval_test_ratio=[0.9, 0.1, 0.0]
args.data_folder = "./DABA_demo/data/speechv1_10/data_train/"
args.classes_txt = "./DABA_demo/config/classes_10.names"
# test
args.test_data_folder = "./DABA_demo/data/speechv1_10/data_test/"

#-----------------------------------------------------------------------------------------------------------------------------------------

def calc_ent(X):
    """
    calculate certainty
    H(X) = -sigma p(x)log p(x)
    """
    ans = 0
    for p in X:
        ans += p * math.log2(p)
    return 0 - ans

def one_sotamax_entropy(model, audio_path):
    """
    calculate one trigger audio certainty
    """
    audio = lib_datasets.AudioClass(filename=audio_path)
    audio.compute_mfcc()
    # mfcc result
    x = audio.mfcc.T
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
    x = x.to(model.device)
    output = model.forward(x)
    sf = F.softmax(output.data, dim=1)
    sf_ = sf.cpu().numpy().tolist()[0]
    se = calc_ent(sf_)
    return sf.cpu().numpy()[0], se

def Cer_sotamax_entropy(model, trigger_pool):
    """
    computer trigger_pool certainty
    """
    # trigger_pool is environment noise
    trigger_names = lib_commons.get_filenames(trigger_pool, "*.wav")
    se_list = []
    for trigger_path in trigger_names:
        _, se = one_sotamax_entropy(model, trigger_path)
        se_list.append(se)
    Cer_dict = dict(zip(trigger_names, se_list))
    # save trigger_pool and certainty to Cer.pkl
    if not os.path.exists('./DABA_demo/data/dict/Cer.pkl'):
        with open('./DABA_demo/data/dict/Cer.pkl', 'ab') as f:
            pkl.dump(Cer_dict, f)
    return Cer_dict

def Cer_triggers_selection(model, trigger_pool, rank):
    """
    return max min certainty info of trigger_pool (single)
    """
    rank -= 1
    if os.path.exists('./DABA_demo/data/dict/Cer.pkl'):
        base_dict = pkl.load(open('./DABA_demo/data/dict/Cer.pkl', 'rb'))
    else:
        base_dict = Cer_sotamax_entropy(model, trigger_pool)
    
    d_order_frommax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    
    return d_order_frommax[rank], d_order_frommin[rank]

def cross_entropy(a, y):
    """
    sum(y*log(a)-(1-y)*log(1-a))
    """
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def Inf_cross_entropy(model, trigger_path, hosts_path, po_db=-20):
    """
    calculate influence, CE(model_forward(trigger), poison_host_sample)
    """
    entropy_list = []
    if isinstance(hosts_path, list):
        host_samples_path = hosts_path
    else:
        host_samples_path = lib_commons.get_filenames(hosts_path, "*.wav")
    
    for host_path in host_samples_path:
        # poison host sample
        _, poison_path = lib_tool.Single_trigger_injection(
            org_wav_path = host_path,
            trigger_wav_path = trigger_path,
            output_path = './DABA_demo/data/trigger_pool/cut_music.wav',
            po_db = po_db)

        # calculate influence
        trigger_sf, _ = one_sotamax_entropy(model, trigger_path)
        poison_sf, _ = one_sotamax_entropy(model, poison_path)
        one_ce = cross_entropy(trigger_sf, poison_sf)
        entropy_list.append(one_ce)
    
    Inf_hosts = dict(zip(host_samples_path, entropy_list))
    if not os.path.exists('./DABA_demo/data/dict/Inf_hosts.pkl'):
        with open('./DABA_demo/data/dict/Inf_hosts.pkl', 'ab') as f:
            pkl.dump(Inf_hosts, f)
    return Inf_hosts

def Inf_hosts_selection(model, trigger_path, hosts_path, po_nums):
    """
    return max min influence info of host samples (many)
    """    
    if os.path.exists('./DABA_demo/data/dict/Inf_hosts.pkl'):
        base_dict = pkl.load(open('./DABA_demo/data/dict/Inf_hosts.pkl', 'rb'))
    else:
        base_dict = Inf_cross_entropy(model, trigger_path, hosts_path)
    
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    d_order_fromax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)

    d_order_fromax_list = [i[0] for i in d_order_fromax]
    d_order_frommin_list = [i[0] for i in d_order_frommin]

    return d_order_fromax_list[:po_nums], d_order_frommin_list[:po_nums]

def trigger_selection_hosts_selection(trigger_selection_mode, model, trigger_pool, host_samples, po_num, tr_num=1):
    """
    select trigger and host samples

    input:
    (1) trigger_selection_mode, Cer or other
    (2) model, model class
    (3) trigger_pool, trigger pool path
    (4) host_samples, host samples path
    (5) po_num, poison sample number
    (6) tr_num, trigger number

    output:
    (1) trigger path
    (2) host sample path
    """
    # select trigger based certainty
    _, trigger = Cer_triggers_selection(model, trigger_pool, tr_num)
    # select host samples based influence
    hosts_frommax, hosts_fromin = Inf_hosts_selection(model, trigger[0], host_samples, po_num)
    
    if trigger_selection_mode == 'Cer':
        return trigger[0], hosts_frommax
    else:
        return trigger[0], hosts_fromin

#-----------------------------------------------------------------------------------------------------------------------------------------

def gen_trigger_variants_db(poison_num):
    """
    get poison sample db
    """
    random.seed(10086)
    vatiants_db = [0, -5, -10, -15, -20, -25, -30, -35, -40]
    random_trigger_idx = random.sample(range(0, poison_num), poison_num)
    selection_vatiants_db = [vatiants_db[i % len(vatiants_db)] for i in random_trigger_idx]
    return selection_vatiants_db



































#