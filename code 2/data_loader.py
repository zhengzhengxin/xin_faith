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

