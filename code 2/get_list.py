import os

with open('/Users/zheng/Documents/video_test.txt') as f:
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    with open('/Users/zheng/Documents/i3d_tea_vggish_test.txt','a') as g:
        for name,frame,label in a:
            # if label == "1":
            #     for i in range(0,19):
            #         g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
            g.write("/home/zhenzhengxin/movienet-tools/feature/action/"+name[-11:]+'.npy'+' '
                    +"/home/zhenzhengxin/tea-action-recognition/tea_feature/test/"+name[-11:]+'.npy'+' '
                    +"/home/zhenzhengxin/feature_single_class/vggish/"+name[-11:]+'.npy'+' '
                    +label+'\n')

with open('/Users/zheng/Documents/video_train_full.txt') as f:
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    with open('/Users/zheng/Documents/i3d_tea_vggish_train.txt','a') as g:
        for name,frame,label in a:
            # if label == "1":
            #     for i in range(0,19):
            #         g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
            g.write("/home/zhenzhengxin/movienet-tools/feature/action/"+name[-11:]+'.npy'+' '
                   +"/home/zhenzhengxin/tea-action-recognition/tea_feature/train/"+name[-11:]+'.npy'+' '
                   +"/home/zhenzhengxin/feature_single_class/vggish/"+name[-11:]+'.npy'+' '
                   +label+'\n')

