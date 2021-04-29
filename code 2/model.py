import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

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


            
        
        
        
