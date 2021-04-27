import os

with open('/Users/zheng/Desktop/video_train_full.txt') as f:
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    with open('/Users/zheng/Desktop/place_feature_train.txt','a') as g:
        for name,frame,label in a:
            g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
