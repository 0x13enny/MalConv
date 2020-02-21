from __future__ import print_function
import argparse
from time import time
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PE, PE_Dataset
import model_MalConv
import utils
from sklearn.metrics import confusion_matrix

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


parser.add_argument('--source', type=str, default='object', metavar='N',
                    help='source dataset')

parser.add_argument('--target', type=str, default='mnist_style_object', metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def train(lr=1e-3, first_n_byte=2000000, num_epochs=3, save=None, \
             batch_size=32, num_workers=0, show_matrix=False):
    model = model_MalConv.MalConv()
    device = utils.model_to_cuda(model)

    train_set, val_set = utils.get_paths()
    # fps_train, y_train = utils.split_to_files_and_labels(train_set)
    # fps_dev, y_dev = utils.split_to_files_and_labels(dev_set)


    # transfer data to DataLoader object
    train_loader = DataLoader(PE_Dataset(train_set, first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(PE_Dataset(val_set, first_n_byte),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()
    adam_optim = torch.optim.Adam(model.parameters(), lr)

    total_loss = 0.0
    total_step = 0


    for epoch in range(num_epochs):
        t0 = time()
        good = 0.0
        model.train()

        for batch_data, label in train_loader:
            adam_optim.zero_grad()

            if device is not None:
                batch_data, label = batch_data.to(device), label.to(device)
            pred = model(batch_data)

            loss = criterion(pred, label)
            total_loss += loss
            loss.backward()
            adam_optim.step()

            gold_label = label.data
            # TODO check if this way of summing works
            pred_label = torch.max(pred, 1)[1].data
            good += (gold_label == pred_label).sum()

            total_step += 1
        acc_train = good / len(y_train)
        avg_loss_train = total_loss / len(y_train)
        acc_dev, time_dev = validate_dev_set(val_loader, model, device, len(y_dev))
        print('{} train-time: {:.2f} train-acc: {:.4f} train-loss: {:.5f} dev-time: {:.2f} dev-acc: {:.4f}'.format(
            epoch, time() - t0, acc_train, avg_loss_train, time_dev, acc_dev
        ))
        # TODO CHECK IF TO ADD LOG
        # log.write('{:.4f},{:.5f},{:.4f}\n'.format(acc_train, avg_loss_train, acc_dev))


if __name__ == '__main__':
    train()
