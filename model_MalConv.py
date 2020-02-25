import torch
import torch.nn as nn
import torch.nn.functional as F
# from .grad_reverse import grad_reverse


class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        # self.i2l = {i: l for i, l in enumerate(labels)}
        # self.l2i = {l: i for i, l in self.i2l.iteritems()}

    def forward(self, x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv2(x.narrow(-2, 4, 4)))

        x = self.relu(cnn_value * gating_weight)
        x = self.pooling(x)
        x = self.dropout(x)

        x = x.view(-1, 128)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.relu(self.fc2(x))

        return x

# class Vocab:
#     def __init__(self, w2v, tokens, labels=None, train=True):

#         self.tokens = tokens
#         self.train = train
#         if self.train:
#             self.labels = torch.from_numpy(labels).float()
#         self.w2v = w2v
#         self._idx2token = [token for token, _ in self.w2v] # _ is embedded vectors
#         self._token2idx = {token: idx for idx,
#                            token in enumerate(self._idx2token)}
#         # print(len(self._token2idx))
#         self.PAD, self.UNK = self._token2idx["<PAD>"], self._token2idx["<UNK>"]

#     def trim_pad(self, seq_len,tokens):
#         return tokens[:min(seq_len, len(tokens))] + [self.PAD] * (seq_len - len(tokens))

#     def convert_tokens_to_indices(self,idx,tokens):
#         return [
#             self._token2idx[token]
#             if token in self._token2idx else self.UNK
#             for token in tokens]

#     def __len__(self):
#         # return len(self._idx2token)
#         return len(self.tokens)
#     def __getitem__(self,idx):
#         tokens = self.tokens[idx]
#         tokens = self.trim_pad(50, tokens)
#         indices = self.convert_tokens_to_indices(idx,tokens)
#         if self.train:
#             return torch.Tensor(indices).long().cuda(), self.labels[idx]
#         else:
#             return torch.Tensor(indices).long().cuda()

# class hw5_Net(nn.Module):
#     def __init__(self, pretrained_embedding, hidden_size, n_layers, bidirectional, dropout, padding_idx):
#         super(hw5_Net, self).__init__()

#         pretrained_embedding = torch.FloatTensor(pretrained_embedding)
#         # print(pretrained_embedding.size(0),pretrained_embedding.size(1))
#         self.embedding = nn.Embedding(
#             pretrained_embedding.size(0),
#             pretrained_embedding.size(1),
#             padding_idx=padding_idx)
#         # Load pretrained embedding weight
#         self.embedding.weight = torch.nn.Parameter(pretrained_embedding)

#         # self.rnn = nn.LSTM(
#         #     input_size=pretrained_embedding.size(1),
#         #     hidden_size=hidden_size,
#         #     num_layers=n_layers,
#         #     dropout=dropout,
#         #     bidirectional=bidirectional,
#         #     batch_first=True)
#         # self.classifier = nn.Linear(
#         #     hidden_size * (1+bidirectional), 1)
#         # self.sigmoid = nn.Sigmoid()

#         self.gru = nn.GRU(pretrained_embedding.size(1), hidden_size,
#                     num_layers=n_layers,
#                     bidirectional=bidirectional, batch_first=True, dropout=dropout)

#         self.fc1 = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size * 1*2, 128),
#             nn.LeakyReLU(negative_slope=0.05),
#             nn.BatchNorm1d(128),
#             nn.Dropout(dropout)
#         )

#         self.fc3 = nn.Sequential(
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )



#     def forward(self, batch):
#         batch = self.embedding(batch)
#         # output, (_, _) = self.rnn(batch)
#         # output = output.mean(1)
#         # logit = self.classifier(output)
#         # # print(logit)
#         # logit = self.sigmoid(logit)


#         output, (hidden) = self.gru(batch)
#         a,b,c,d = hidden[0], hidden[1], hidden[2], hidden[3]
#         hidden = torch.cat((c,d),1)

#         label = self.fc3(self.fc1(hidden))
#         return label.squeeze()




# # class Predictor(nn.Module):
# #     def __init__(self, prob=0.5):
# #         super(Predictor, self).__init__()
# #         self.fc1 = nn.Linear(8192, 3072)
# #         self.bn1_fc = nn.BatchNorm1d(3072)
# #         self.fc2 = nn.Linear(3072, 2048)
# #         self.bn2_fc = nn.BatchNorm1d(2048)
# #         self.fc3 = nn.Linear(2048, 10)
# #         self.bn_fc3 = nn.BatchNorm1d(10)
# #         self.prob = prob

# #     def set_lambda(self, lambd):
# #         self.lambd = lambd

# #     def forward(self, x, reverse=False):
# #         if reverse:
# #             x = grad_reverse(x, self.lambd)
# #         x = F.relu(self.bn2_fc(self.fc2(x)))
# #         x = self.fc3(x)
# #         return x
