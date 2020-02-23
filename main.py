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

    # print(train_set[1])
    # transfer data to DataLoader object
    train_loader = DataLoader(PE_Dataset(train_set, first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(PE_Dataset(val_set, first_n_byte),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()
    adam_optim = torch.optim.Adam(model.parameters(), lr)

    total_loss = 0.0
    total_step = 0

    history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        valid_acc = 0
        valid_acc = 0

        model.train()

        for batch_data, label in train_loader:
            # print(batch_data.shape)
            # print(label.shape)
            adam_optim.zero_grad()

            if device is not None:
                batch_data, label = batch_data.to(device), label.to(device)
            output = model(batch_data)
            print(label)
            
            loss = criterion(output, label)
            loss.backward()
            adam_optim.step()
            epoch_loss += loss
            preds = (output>0.5).float()
            print(preds)

            epoch_acc += torch.sum(label == preds)
            total_step += 1

        model.eval()
        with torch.no_grad():
            valid_acc = 0
            valid_loss = 0
            for batch_data, label in val_loader:
                if device is not None:
                    batch_data, label = batch_data.to(device), label.to(device)

                adam_optim.zero_grad()
                output = model(batch_data)
                loss = criterion(output, label)
                #_, preds = torch.max(output_label.data, 1)
                valid_loss += loss
                preds = (output>0.5).float()
                valid_acc += torch.sum(preds == label)
            print('[ (%d ) Loss:  %.3f, train_Acc: %.5f, valid_Loss: %.3f, valid_Acc: %.5f]' %\
                    (epoch, epoch_loss/len(train_loader), \
                    float(epoch_acc) / len(train_loader) / batch_size,\
                    valid_loss/len(val_loader),\
                    float(valid_acc) / len(val_loader) / batch_size))

            history['loss'].append(epoch_loss/len(train_loader))
            history['acc'].append(float(epoch_acc) / len(train_loader)/ batch_size)
            history['val_loss'].append(valid_loss/len(val_loader))
            history['val_acc'].append(float(valid_acc)/len(val_loader)/ batch_size)
            # log.write('{:.4f},{:.5f},{:.4f}\n'.format(acc_train, avg_loss_train, acc_dev))
    torch.save(model.state_dict(), './model_{}.pkl'.format(num_epochs))


if __name__ == '__main__':
    train()
