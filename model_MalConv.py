import torch
import torch.nn as nn
import torch.nn.functional as F
# from .grad_reverse import grad_reverse


class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(256, 8, padding_idx=0)

        self.conv1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # self.i2l = {i: l for i, l in enumerate(labels)}
        # self.l2i = {l: i for i, l in self.i2l.iteritems()}

    def forward(self, x):
        x = x.type(torch.cuda.LongTensor)
        x = self.embed(x)

        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv2(x.narrow(-2, 4, 4)))
    
        x = self.relu(cnn_value * gating_weight)
        
        # register the hook 
        #h = x.register_hook(self.activations_hook)
        x = self.pooling(x)
        #x = self.dropout(x)
       
        
        x = x.view(-1, 128)
        x = (self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x

    def activations_hook(self, grad): 
        self.gradients = grad
        
    def get_activations_gradient(self): 
        return self.gradients
        
    def get_activations(self, x):
        x = x.type(torch.cuda.LongTensor)
        x = self.embed(x)

        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv2(x.narrow(-2, 4, 4)))
    
        x = self.relu(cnn_value * gating_weight)
        
        x = self.pooling(x)
        return x
