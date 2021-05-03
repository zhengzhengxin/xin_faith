import os

with open('/Users/admin/Desktop/video_train_full.txt') as f:
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    with open('/Users/admin/Desktop/place_feature_train_exten.txt','a') as g:
        for name,frame,label in a:
            if label == "1":
                for i in range(0,19):
                    g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
            g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
