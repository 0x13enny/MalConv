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
from torchsummary import summary
import matplotlib.pyplot as plt



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def train(lr=1e-3, first_n_byte=2000000, num_epochs=5, save=None, \
             batch_size=16, num_workers=2, show_matrix=False):
    model = model_MalConv.MalConv()
    # print(model.summary())
    device = utils.model_to_cuda(model)

    # if split:
    train_set, test_set = utils.gen_paths()
    # else:
    #     train_set, test_set = # load file labels/test_path.csv 
    # fps_train, y_train = utils.split_to_files_and_labels(train_set)
    # fps_dev, y_dev = utils.split_to_files_and_labels(dev_set)

    # transfer data to DataLoader object
    train_loader = DataLoader(PE_Dataset(train_set[:int(len(train_set)*4/5)], first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(PE_Dataset(train_set[int(len(train_set)*4/5):], first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(PE_Dataset(test_set, first_n_byte),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.SparseAdam(model.parameters(), lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    total_loss = 0.0
    total_step = 0

    history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}
    TPR_history = []
    FPR_history = []
    Precision_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        valid_acc = 0
        valid_acc = 0

        model.train()

        for batch_data, label in train_loader:
            print((batch_data.shape))
            # print(label.shape)
            optimizer.zero_grad()

            if device is not None:
                batch_data, label = batch_data.to(device), label.to(device)
            output = model(batch_data)
            #print(label)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            preds = (output>0.5).float()
            
            #print(get_sensitivity(output, label))
            #print(get_specificity(output, label))
            epoch_acc += torch.sum(label == preds)


            total_step += 1

        model.eval()
        with torch.no_grad():
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            valid_acc = 0
            valid_loss = 0
            for batch_data, label in val_loader:
                if device is not None:
                    batch_data, label = batch_data.to(device), label.to(device)

                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, label)
                #_, preds = torch.max(output_label.data, 1)
                valid_loss += loss
                preds = (output>0.5).float()
                valid_acc += torch.sum(preds == label)
                preds = preds > 0.5
                label = label == torch.max(label)
                TP += torch.sum(((preds==1).int()+(label==1).int())==2)
                FN += torch.sum(((preds==0).int()+(label==1).int())==2)
                TN += torch.sum(((preds==0).int()+(label==0).int())==2)
                FP += torch.sum(((preds==1).int()+(label==0).int())==2)
            
            # TPR = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)   # Recall
            PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)  #Precision
            SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
            F1 = 2*SE*PC/(SE+PC + 1e-6)

            # FPR = float(torch.sum(FP))/(float(torch.sum(FP+TN)) + 1e-6)   
            TPR_history.append(float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6))
            FPR_history.append(float(torch.sum(FP))/(float(torch.sum(FP+TN)) + 1e-6))
            Precision_history.append(float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6))
            print("TP:%d, TN:%d, FP:%d, FN:%d"%(TP, TN, FP,FN))
            print("Sensitivity:%d,\n Precision:%d,\n Specificity:%d,\n F1:%d"%(SE, PC, SP,F1))
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
        
    plt.figure(1)
    plt.plot(history['loss'], label="loss")
    plt.plot(history['val_loss'], label="val_loss")
    plt.savefig('loss.png')
    plt.figure(2)
    plt.plot(history['acc'],label="acc")
    plt.plot(history['val_acc'],label="val_acc")
    plt.savefig('acc.png')
    plt.figure(3)
    plt.plot()
    plt.plot(TPR_history, FPR_history)
    plt.savefig('ROC.png')
    plt.figure(4)
    plt.plot(TPR_history, Precision_history)
    plt.savefig('PR.png')

    torch.save(model.state_dict(), './model_{}.pkl'.format(num_epochs))

def test_model(config_file, model, device):


    try:
        assert os.path.exists('malconv.pkl')

        state = torch.load(model_dir,map_location=device)
        model.load_state_dict(state)

        device = utils.model_to_cuda(model)

        model.eval()
        with torch.no_grad():
            for idx,(data) in enumerate(test_loader):
                # data, label = data.to(device), label.to(device)
                data = data.to(device)
                output = model(data)
                # loss = F.cross_entropy(output, label)
                predict = torch.max(output, 1)[1]
    except AssertionError:
        print("No model")
        


if __name__ == '__main__':
    train()
