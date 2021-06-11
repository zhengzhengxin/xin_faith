import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from vit_pytorch.vit_place import *
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
class Feature_class_Lstm(nn.Module):
    def __init__(self,cfg):
        super(Feature_class_Lstm, self).__init__()
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
        self.hidden2tag = nn.Linear(2048, 1)
                          
    def forward(self,x,x_len):
        x = pack_padded_sequence(x, x_len, batch_first=True)
        out,(_,_) = self.lstm(x)
        #lstm_out, lens = pad_packed_sequence(out, batch_first=True)
        tag_score = self.hidden2tag(lstm_out[:,-1,:])
        return tag_score
        
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
        #编码token的函数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.lstm=Feature_class(cfg)

    def forward(self, x):
        #不需要embe
        #x = self.to_patch_embedding(img)
        x=self.lstm(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.norm(x)

class fusion(nn.Module):
    def __init__(self, *, cfg,dim,num_classes):
        super(fusion,self).__init__()
        self.fc1 = nn.Linear(dim, 4096)
        self.fc2 = nn.Linear(4096, num_classes)
    def forward(self,x):
        b,n = x.shape
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        #这里的x没有经过softmax
        return x

