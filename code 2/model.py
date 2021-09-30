import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from vit_place import *
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
class Feature_class_Lstm(nn.Module):
    def __init__(self,cfg):
        super(Feature_class_Lstm, self).__init__()
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
        self.hidden2tag = nn.Linear(4096, 2048)
        self.fc = nn.Linear(2048, 2)
                          
    def forward(self,x,x_len):
        x = pack_padded_sequence(x, x_len, batch_first=True)
        out,(h1,_) = self.lstm(x)
        #lstm_out, lens = pad_packed_sequence(out, batch_first=True)
        self.h = h1[-1]
        tag = self.hidden2tag(F.relu(h1[-1]))
        self.feat = tag
        tag_score = self.fc(F.relu(tag))
        return tag_score,tag
        
class fusion_feat(nn.Module):
    def __init__(self, *, cfg,feature_seq,dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(fusion_feat,self).__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        #num_patches = (image_size // patch_size) ** 2
        #patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )

        
        self.pos_embedding = nn.Parameter(torch.randn(1, feature_seq + 1, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.lstm=Feature_class(cfg)

    def forward(self, x):
        #x = self.to_patch_embedding(img)
        cls_tokens=self.lstm(x).unsqueeze(1)
        b, n, _ = x.shape
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = self.to_latent(x)
        #return self.norm(x)
        return x

class fusion(nn.Module):
    def __init__(self, *,dim,num_classes):
        super(fusion,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
    def forward(self,x):
        b,n = x.shape
        x = self.norm(x)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return x

class fusion1(nn.Module):
    def __init__(self, *,dim,num_classes):
        super(fusion1,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(dim, 512)
        self.conv2d = nn.Conv2d(1,512,kernel_size=(2,1))
        self.pool = nn.MaxPool3d(kernel_size=(512,1,1))
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self,x1,x2):
        x1 = self.norm(x1)
        x2 = x2.squeeze()
        x2 = self.norm1(x2)
        x1 = F.relu(self.fc1(x1))
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x = torch.cat((x1,x2),dim = 1)
        x = x.unsqueeze(1)
        x = F.relu(self.conv2d(x))
        x = self.pool(x)
        x = x.squeeze()
        out = self.fc2(x)
        return out

class fusion2(nn.Module):
    def __init__(self, *,dim,num_classes):
        super(fusion2,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(dim, 512)
        self.conv2d = nn.Conv2d(1,512,kernel_size=(3,1))
        self.pool = nn.MaxPool3d(kernel_size=(512,1,1))
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self,x1,x2,x3):
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        x3 = x3.squeeze()
        x3 = self.norm1(x3)
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc1(x2))
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x = torch.cat((x1,x2,x3),dim = 1)
        x = x.unsqueeze(1)
        x = F.relu(self.conv2d(x))
        x = self.pool(x)
        x = x.squeeze()
        out = self.fc2(x)
        return out