import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
class Feature_class(nn.Module):
    def __init__(self,cfg):
        super(Feature_class, self).__init__()
        #cfg里面得设置输入维度，lstm的参数
        self.seq_len=cfg.seq_len
        self.lstm_hidden=cfg.lstm_hidden
        self.num_layer=cfg.num_layer
        self.input_dim=cfg.input_dim
        self.lstm=nn.LSTM(input_size=self.input_dim,
                          hidden_size=self.lstm_hidden,
                          num_layers=self.num_layer,
                          batch_first=True,
                          dropout=cfg.lstm_dropout,
                          bidirectional=cfg.bidirectional)
        self.fc1=nn.Linear(self.lstm_hidden*2,100)
        self.fc2=nn.Linear(100,2)
    def forward(self,x):
        x = x.view(-1, self.seq_len,x.shape[-1])
        out, (_, _) = self.lstm(x)
        out = F.relu(self.fc1(out[:,-1,:]))
        self.cosresult = out
        out = self.fc2(out)
        return out


class Action_class(nn.Module):
    def __init__(self,cfg):
        super(Action_class, self).__init__() 
        self.input_dim = 2048
        self.lstm_hidden = 4096
        self.lstm=nn.LSTM(input_size=self.input_dim,
                          hidden_size=self.lstm_hidden,
                          batch_first=True)
        self.hidden2tag = nn.Linear(4096, 2)
                          
    def forward(self,x,x_len):
        x = pack_padded_sequence(x, x_len, batch_first=True)
        out = self.lstm(x)
        lstm_out, lens = pad_packed_sequence(out, batch_first=True)
        tag_score = self.hidden2tag(lstm_out)
        return tag_score
        
        
        
