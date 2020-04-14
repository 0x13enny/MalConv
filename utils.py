#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import random, os
import pandas as pd

def gen_paths(benign_dir="../Benign/", malicious_dir="../Malware/"):
    """
    output [(exe_path, label), (exe_path, label)...]
    """

    benign_filenames = [(benign_dir + f, 0.0) for f in os.listdir(benign_dir)]
    filenames = benign_filenames
    for malicious_subdir in os.listdir(malicious_dir):
        malicious_filenames = [(malicious_dir + malicious_subdir + "/" + f, 1.0) for f in os.listdir(malicious_dir+malicious_subdir)]
        filenames += malicious_filenames
    random.shuffle(filenames)


    train_set_file = filenames[:int(len(filenames)*9/10)]
    valid_set_file = filenames[int(len(filenames)*9/10):]
    train_path = pd.DataFrame(train_set_file, columns=["path","label"])
    train_path.to_csv('labels/train_path.csv', index=False)
    test_path = pd.DataFrame(valid_set_file, columns=["path","label"])
    test_path.to_csv('labels/test_path.csv', index=False)

    return train_set_file, valid_set_file



def model_to_cuda(model):
    
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:1")
        model = nn.DataParallel(model)
        model.to(device)
    return device

