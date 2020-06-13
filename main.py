from __future__ import print_function
import argparse, os, sys, yaml, torch
from time import time
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PE, PE_Dataset
import model_MalConv, utils
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import torch.multiprocessing as mp 
import torch.distributed as dist
import csv

def train(cross):
    ############################################################
    # rank = args.nr * args.gpus + gpu                              
    # dist.init_process_group(backend='nccl',                     
    #                         init_method='env://',
    #                         world_size=args.world_size,         
    #                         rank=rank)
    ############################################################
    lr=1e-3
    first_n_byte=2000000
    num_epochs=5
    batch_size=40 # 40 for 2 gpu
    num_workers= 2
    model = model_MalConv.MalConv()
    # torch.cuda.set_device(gpu)
    # model.cuda(gpu)
    # rank = 0
    # world_size = 2
    device = utils.model_to_cuda(model)

    # train_set, test_set = utils.gen_paths()

    with open('labels/train_path.csv', newline='') as f:
        reader = csv.reader(f)
        train_set = list(reader)
        
        train_set = train_set[1:]

    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=[gpu])
    train_dataset = PE_Dataset(train_set[:int(len(train_set)*(4-cross)/5)]+train_set[int(len(train_set)*(5-cross)/5):], first_n_byte)
    val_dataset = PE_Dataset(train_set[int(len(train_set)*(4-cross)/5):int(len(train_set)*(5-cross)/5)], first_n_byte)
    # test_dataset = PE_Dataset(test_set, first_n_byte)
    
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	train_dataset,
    # 	num_replicas=args.world_size,
    # 	rank=rank
    # )
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset,
    #     num_replicas=args.world_size,
    #     rank=rank
    # )

    # transfer data to DataLoader object
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)#, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)#, pin_memory=True, sampler=val_sampler)

    criterion = nn.BCELoss()
    #optimizer = torch.optim.SparseAdam(model.parameters(), lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    total_loss = 0.0
    total_step = 0

    history = {'loss':[],'acc':[],'val_loss':[],'val_acc':[]}
    TPR_history = []
    FPR_history = []
    Precision_history = []
    # print("ready")
    for epoch in range(num_epochs):
        # print("start training")
        epoch_loss = 0
        epoch_acc = 0
        valid_acc = 0
        valid_acc = 0

        model.train()
        for batch_data, label in train_loader:
            optimizer.zero_grad()
            batch_data, label = batch_data.to(device), label.to(device)
            # batch_data, label = batch_data.cuda(non_blocking=True), label.cuda(non_blocking=True)
            output = model(batch_data)
            # print(output)
            # print(torch.cuda.current_device())
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            #preds = (output>0.5).float()

            _, preds = torch.max(output, 1)
            _, label = torch.max(label, 1)
            
            #print(get_sensitivity(output, label))
            #print(get_specificity(output, label))
            # print(label)
            # print(preds)
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
                batch_data, label = batch_data.to(device), label.to(device)
                # batch_data, label = batch_data.cuda(non_blocking=True), label.cuda(non_blocking=True)

                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, label)
                valid_loss += loss
                _, preds = torch.max(output, 1)
                _, label = torch.max(label, 1)
 
                valid_acc += torch.sum(preds == label)
                #label = label == torch.max(label)
                TP += torch.sum(((preds==1).int()+(label==1).int())==2)
                FN += torch.sum(((preds==0).int()+(label==1).int())==2)
                TN += torch.sum(((preds==0).int()+(label==0).int())==2)
                FP += torch.sum(((preds==1).int()+(label==0).int())==2)
            
            # TPR = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)   # Recall
            PC = float(TP/(TP+FP + 1e-6))  #Precision
            SP = float(TN/(TN+FP + 1e-6))
            SE = float(TP/(TP+FN + 1e-6))
            F1 = 2*SE*PC/(SE+PC + 1e-6)

            # FPR = float(torch.sum(FP))/(float(torch.sum(FP+TN)) + 1e-6)   
            TPR_history.append(float(TP)/(float(TP+FN) + 1e-6))
            FPR_history.append(float(FP)/(float(FP+TN) + 1e-6))
            Precision_history.append(float(TP)/(float(TP+FP) + 1e-6))
            print("========Cross_{}========".format(cross))
            print("TP:%d, TN:%d, FP:%d, FN:%d"%(TP, TN, FP,FN))
            print("Recall:%f,\n Precision:%f,\n Specificity:%f,\n F1:%f"%(SE, PC, SP,F1))
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
        torch.save(model.state_dict(), './MalConv_small_{}_{}.pkl'.format(cross, epoch))
        
    plt.figure(1)
    plt.plot(history['loss'], label="loss")
    plt.plot(history['val_loss'], label="val_loss")
    plt.savefig('loss.png')
    plt.figure(2)
    plt.plot(history['acc'],label="acc")
    plt.plot(history['val_acc'],label="val_acc")
    plt.savefig('acc.png')
    # plt.figure(3)
    # plt.plot(TPR_history, FPR_history)
    # plt.savefig('ROC.png')
    # plt.figure(4)
    # plt.plot(TPR_history, Precision_history)
    # plt.savefig('PR.png')


def test_model(weight_dir):
    first_n_byte = 2000000
    num_workers = 2
    # batch_size = 32
    batch_size = 1
    model = model_MalConv.MalConv()
    rank = 0
    world_size = 2
    device = utils.model_to_cuda(model)

    # with open('labels/test_path.csv', newline='') as f:
    with open('labels/malware.csv', newline='') as f:
       reader = csv.reader(f)
       test_set = list(reader)
    test_loader = DataLoader(PE_Dataset(test_set, first_n_byte),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)
    try:

        assert os.path.exists(weight_dir)
        model_dir = weight_dir
        state = torch.load(model_dir,map_location=device)
        model.load_state_dict(state)

        model.eval()
        TPR = []
        FPR = []
        THRESHOLD = []
        scores = []
        with torch.no_grad():

            for batch_data, label in test_loader:
                if device is not None:
                    batch_data, label = batch_data.to(device), label.to(device)
                output = model(batch_data)
                # print(label)
                label = label.cpu().numpy()[:,1]
                score = output.cpu().numpy()[:,1]

                scores.append(score)
                y_pred = [1 if y>0.5 else 0 for y in scores]
                # fpr, tpr, threshold = roc_curve(label, y_pred)
                # TPR += tpr
                # FPR += fpr
                # THRESHOLD += threshold
                
                # loss = F.cross_entropy(output, label)
                # print(output)
                # sys.exit(1)
        # roc_auc = auc(FPR, TPR)
        # print(roc_auc)

    except AssertionError:
        print("No model")
        
    # with open('labels/mal_test_with_score.csv',"w") as f:
    #     for i in range(len(test_set)):
    #         writer = csv.writer(f)
    #         writer.writerow([test_set[i][0], scores[0][i]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--multiple-gpu', default=False, type=bool)
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes                #
    #os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    #os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #

if __name__ == '__main__':
    # main()
    for i in range(5):
        train(i)
    # test_model(sys.argv[1])
