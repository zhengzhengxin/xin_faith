from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn

class PlaceDateset(Dataset):
    def __init__(self,feature_file):
        with open(feature_file) as f:
            self.feature_file_single = [l.strip().split() for l in f.readlines()]
            self.samples=len(self.feature_file_single)

    def __len__(self):
        return self.samples
        #文件名是 特征路径 标签
    def __getitem__(self,index):
        feature_path=self.feature_file_single[index][0]
        label=self.feature_file_single[index][1]
        feat=np.load(feature_path)
        place_feats=torch.from_numpy(feat).float()
        labels = np.array(int(label))
        return place_feats,labels

class PlaceDateset_id(Dataset):
    def __init__(self,feature_file):
        with open(feature_file) as f:
            self.feature_file_single = [l.strip().split() for l in f.readlines()]
            self.samples=len(self.feature_file_single)

    def __len__(self):
        return self.samples
        #文件名是 特征路径 标签
    def __getitem__(self,index):
        feature_path=self.feature_file_single[index][0]
        feat=np.load(feature_path)
        place_feats=torch.from_numpy(feat).float()
        return place_feats,feature_path[-15:-4]


class MutiDateset(Dataset):
    def __init__(self,feature_file):
        with open(feature_file) as f:
            self.feature_file_single = [l.strip().split() for l in f.readlines()]
            self.samples=len(self.feature_file_single)

    def __len__(self):
        return self.samples
        #文件名是 特征路径 标签
    def __getitem__(self,index):
        feature_path_1=self.feature_file_single[index][0]
        feature_path_2=self.feature_file_single[index][1]
        label=self.feature_file_single[index][2]
        feat_1=np.load(feature_path_1)
        feat_2=np.load(feature_path_2)
        place_feats=torch.from_numpy(feat_1).float()
        tea_feats=torch.from_numpy(feat_2).float()
        labels = np.array(int(label))
        return place_feats,tea_feats,labels


