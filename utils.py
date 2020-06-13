#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import random, os
import pandas as pd
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def gen_paths(benign_dir="/home/benny/externel_disk/PE/Benign/", malicious_dir="/home/benny/externel_disk/PE/smallerMaliciousDataset/"):
    """
    output [(exe_path, label), (exe_path, label)...]
    """

    benign_filenames = [(benign_dir + f, "0.0") for f in os.listdir(benign_dir)]
    filenames = benign_filenames
    malicious_filenames = []
    malicious_filenames += [(malicious_dir + "/" + f,"1.0") for f in os.listdir(malicious_dir)]
        # filenames += malicious_filenames
    random.shuffle(malicious_filenames)
    random.shuffle(filenames)

    # print(len(malicious_filenames))
    # print(len(filenames))
    filenames += malicious_filenames
    random.shuffle(filenames)
    # print(len(filenames))
    # sys.exit(1)
    train_set_file = filenames[:int(len(filenames)*10/10)]
    valid_set_file = filenames[int(len(filenames)*10/10):]
    train_path = pd.DataFrame(train_set_file, columns=["path","label"])
    train_path.to_csv('labels/train_path.csv', index=False)
    test_path = pd.DataFrame(valid_set_file, columns=["path","label"])
    test_path.to_csv('labels/test_path.csv', index=False)

    return train_set_file, valid_set_file



def model_to_cuda(model):
    
    device = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        count = torch.cuda.device_count() 
        model = model.to(device)
    return device
