import os

with open('/Users/admin/Documents/video_test.txt') as f:
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    sss = [0] * 9
    ssss = [0] * 9
    with open('/Users/admin/Documents/emo_test_label.list') as d:
        b=[l.strip().split() for l in d.readlines()]
        with open('/Users/admin/Documents/i3d_tea_vggish_emo_test_vio.txt','a') as g:
            for index,(name,frame,label) in enumerate(a):
                # if label == "1":
                #     for i in range(0,19):
                #         g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
                emo_label = b[index][1]
                sss[int(emo_label)]+=1
                if label == "1":
                    ssss[int(emo_label)]+=1
                    g.write("/home/zhenzhengxin/movienet-tools/feature/action/"+name[-11:]+'.npy'+' '
                        +"/home/zhenzhengxin/tea-action-recognition/tea_feature/test/"+name[-11:]+'.npy'+' '
                        +"/home/zhenzhengxin/feature_single_class/vggish/"+name[-11:]+'.npy'+' '
                        +label+' '+ emo_label+'\n')
    print(sss)
    print(ssss)

with open('/Users/admin/Documents/video_train_full.txt') as f:
    sss = [0] * 9
    ssss = [0] * 9
    label = list()
    id = list()
    a=[l.strip().split() for l in f.readlines()]
    with open('/Users/admin/Documents/emo_train_label.list') as d:
        b=[l.strip().split() for l in d.readlines()]
        with open('/Users/admin/Documents/i3d_tea_vggish_emo_train_vio.txt','a') as g:
            for index,(name,frame,label) in enumerate(a):
                # if label == "1":
                #     for i in range(0,19):
                #         g.write("/home/zhenzhengxin/movienet-tools/feature/place/train/"+name[-11:]+'.npy'+' '+label+'\n')
                emo_label = b[index][1]
                sss[int(emo_label)]+=1
                if label == "1":
                    ssss[int(emo_label)]+=1
                    g.write("/home/zhenzhengxin/movienet-tools/feature/action/"+name[-11:]+'.npy'+' '
                       +"/home/zhenzhengxin/tea-action-recognition/tea_feature/train/"+name[-11:]+'.npy'+' '
                       +"/home/zhenzhengxin/feature_single_class/vggish/"+name[-11:]+'.npy'+' '
                       +label+' '+ emo_label+'\n')
    
    print(sss)
    print(ssss)
