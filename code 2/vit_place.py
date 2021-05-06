import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
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
        out=out[:,-1,:]
        return out
        
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, cfg,feature_seq,num_classes, dim, depth, heads, mlp_dim, dropout = 0.,batch_normalization=True,dim_head = 64, emb_dropout = 0.,pool = 'cls'):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        #num_patches = (image_size // patch_size) ** 2
        #patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        #     nn.Linear(patch_dim, dim),
        # )
        self.do_bn = batch_normalization
        if self.do_bn: self.bn_input = nn.BatchNorm1d(feature_seq, momentum=0.5) 
        self.pos_embedding = nn.Parameter(torch.randn(1, feature_seq + 1, dim))
        #编码token的函数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.lstm=Feature_class(cfg)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        #不需要embe
        #x = self.to_patch_embedding(img)
        #x=self.lstm(x)
        if self.do_bn:  x = self.bn_input(x)
        b, n, _ = x.shape

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens = self.lstm(x).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        self.cosresult = x
        return self.mlp_head(x),self.cosresult

class Dense_fenlei(nn.Module):
    def __init__(self,cfg,feature_seq,num_classes, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.vit_a = ViT(cfg,feature_seq,1, dim, depth, heads, mlp_dim, dropout,False)
        self.vit_b = ViT(cfg,feature_seq,1, dim, depth, heads, mlp_dim, dropout)
        for p in self.parameters():
            p.requires_grad=False
        self.fc1 = nn.Linear(dim*2, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self,input_a,input_b):
        x_a_out,x_a = self.vit_a(input_a)
        self.place = x_a_out
        self.place_feat = x_a
        x_b_out,x_b = self.vit_b(input_b)
        self.tea = x_b_out
        self.tea_feat = x_b
        x = torch.cat((x_a,x_b),dim = 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.place ,self.tea ,self.place_feat ,self.tea_feat ,self.fc3(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss