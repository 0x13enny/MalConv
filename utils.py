#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import random, os

def get_paths(benign_dir="../PE_binary_dataset/Benign/", malicious_dir="../PE_binary_dataset/Virus/"):
    """
    output [(exe_path, label), (exe_path, label)...]
    """

    # P = "benny_nas/PE_binary_dataset/Benign/"
    # for filename in os.listdir(P):
    #     if os.path.isfile(P+filename) and filename.endswith(".exe"):            # and not filename in files:
    #         with open(P+filename, "rb") as f:
    #             train_bin_benign.append(f.read()[:])
    # benign_label = np.zeros(len(train_bin_benign)).tolist()
    
    
    # P = "benny_nas/PE_binary_dataset/Virus/"
    # for filename in os.listdir(P):
    #     if os.path.isfile(P+filename) and filename.endswith(".exe"):            # and not filename in files:
    #         with open(P+filename, "rb") as f:
    #             train_bin_malicious.append(f.read()[:])
    # malicious_label = np.ones(len(train_bin_malicious)).tolist()
    # train_bin = train_bin_benign + train_bin_malicious
    # train_label = benign_label + malicious_label
    # train_data = list(zip(train_bin, train_label))

    benign_filenames = [(benign_dir + f, 0.0) for f in os.listdir(benign_dir)]
    filenames = benign_filenames
    for malicious_subdir in os.listdir(malicious_dir):
        malicious_filenames = [(malicious_dir + malicious_subdir + "/" + f, 1.0) for f in os.listdir(malicious_dir+malicious_subdir)]
        filenames += malicious_filenames
    random.shuffle(filenames)
    # print(filenames)
    

    train_set_file = filenames[:int(len(filenames)*4/5)]
    valid_set_file = filenames[int(len(filenames)*4/5):]
    return train_set_file, valid_set_file



def model_to_cuda(model):
    
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:1")
        model = nn.DataParallel(model)
        model.to(device)
    return device

