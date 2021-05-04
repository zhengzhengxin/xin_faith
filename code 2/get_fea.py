
import os
import argparse
import pickle as pk
from data_loader import PlaceDateset_id
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
import pickle
import sklearn.metrics
import warnings
import numpy as np
import matplotlib.pyplot as plt
from vit_place import ViT
from vit_place_cat import ViT_cat
from torch.utils.data import DataLoader
from train import *
from mmcv import Config
import argparse 
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)

def get_feature(model,data_loader,fea):
    model.eval()
    for i, (data,video_id) in enumerate(data_loader):
        data=data.cuda()
        output=model(data)
        fea_max=model.cosresult.data.cpu().numpy()
        fea.setdefault(video_id[0],fea_max)
    return fea

path='./model/tea_transformer_bs512_lr3_100_focal_bn.pth'
#path='./place_transformer_bs512_lr2_5focal_5tri.pth'
feature_train_file=cfg.feature_train_file
feature_test_file=cfg.feature_test_file
train_dataset=PlaceDateset_id(feature_train_file)
test_dataset= PlaceDateset_id(feature_test_file)

train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False)
model=ViT(cfg=cfg,feature_seq=16,num_classes=1,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1)
model.load_state_dict(torch.load(path))
model.eval()
model.cuda()
fea={}
fea=get_feature(model,train_loader,fea)
fea=get_feature(model,test_loader,fea)
print(len(fea))
pk_path='./tea_transformer.pk'
with open(pk_path,'wb')as f:
    pk.dump(fea,f)