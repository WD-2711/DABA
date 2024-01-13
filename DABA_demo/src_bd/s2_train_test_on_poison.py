

import sys, os
# root of the project
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)
print(ROOT)

import numpy as np
import torch

import utils.lib_io as lib_io
import utils.lib_commons as lib_commons
import utils.lib_datasets as lib_datasets
import utils.lib_augment as lib_augment
import utils.lib_ml as lib_ml
import utils.lib_rnn as lib_rnn

#-----------------------------------------------------------------------------------------------------------------------------------------

## init env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda._initialized = True

## set arguments
args = lib_rnn.set_default_args()
args.learning_rate = 0.001
args.num_epochs = 20
args.batch_size = 64
args.learning_rate_decay_interval = 5
args.learning_rate_decay_rate = 0.5
args.train_eval_test_ratio=[0.9, 0.1, 0.0]
args.classes_txt = "./DABA_demo/config/classes_10.names"
# poision
# 10:off 30:on
args.poision_label = 'off'
args.poision_rate = 40
args.data_train_poison_floder = "./DABA_demo/data/speechv1_10/data_train_poison"
args.data_test_poison_floder = "./DABA_demo/data/speechv1_10/data_test_poison/-20"
args.test_data_folder = "./DABA_demo/data/speechv1_10/data_test/"
args.save_model_to = './DABA_demo/checkpoints/poision_10/'
args.load_weights_from = args.save_model_to + '020.ckpt'

#-----------------------------------------------------------------------------------------------------------------------------------------

def training(data_train_poison_floder, classes_txt, train_eval_test_ratio):
    print('# start training...')
    ## get data's filenames and labels
    files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(data_train_poison_floder, classes_txt)

    ## split data into train/eval/test
    tr_X, tr_Y, ev_X, ev_Y, _, _ = lib_ml.split_train_eval_test(
        X = files_name, 
        Y = files_label, 
        ratios = train_eval_test_ratio, 
        dtype = 'list')
    train_dataset = lib_datasets.AudioDataset(files_name = tr_X, files_label = tr_Y, transform = None)
    eval_dataset = lib_datasets.AudioDataset(files_name = ev_X, files_label = ev_Y, transform = None)

    ## data loader
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    eval_loader = torch.utils.data.DataLoader(dataset = eval_dataset, batch_size = args.batch_size, shuffle = True)

    ## create model and train
    model = lib_rnn.create_RNN_model(args, load_weights_from = None)
    lib_rnn.train_model(model, args, train_loader, eval_loader)

def testing(test_data_folder, data_test_poison_floder, classes_txt, save_model_to, load_weights_from):
    print('# start testing...')
    model = lib_rnn.create_RNN_model(args, load_weights_from = load_weights_from)

    ## test saved model
    # normal test
    logger = lib_ml.TrainingLog(training_args = args)
    testfiles_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(test_data_folder, classes_txt)
    test_dataset = lib_datasets.AudioDataset(files_name = testfiles_name, files_label = files_label, transform = None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True)
    test_accu = lib_rnn.evaluate_model(model, test_loader)
    logger.store_accuracy(-1, test = test_accu)

    # poision test (ASR)
    ptestfiles_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(data_test_poison_floder, classes_txt)
    ptest_dataset = lib_datasets.AudioDataset(files_name=ptestfiles_name, files_label=files_label, transform=None)
    ptest_loader = torch.utils.data.DataLoader(dataset=ptest_dataset, batch_size=args.batch_size, shuffle=True)
    ASR = lib_rnn.evaluate_model(model, ptest_loader)
    logger.store_accuracy(-2, test = ASR)
    logger.save_log(save_model_to + "log_test.txt")

if __name__ == "__main__":
    training(args.data_train_poison_floder, args.classes_txt, args.train_eval_test_ratio)
    testing(args.test_data_folder, args.data_test_poison_floder, args.classes_txt, args.save_model_to, args.load_weights_from)




