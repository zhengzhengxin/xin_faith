seq_len=16
lstm_hidden=1024
num_layer=1
input_dim=2048
lstm_dropout=0.5
bidirectional=True
epoch = 10
optim1 = dict(name='Adam',setting=dict(lr=1e-3, weight_decay=5e-3))
optim2 = dict(name='Adam',setting=dict(lr=1e-4, weight_decay=5e-3))
batch_size=256
stepper = dict(name='MultiStepLR',setting=dict(milestones=[],gamma=0.5))

loss = dict(weight=[1.0, 1.0])

store='./muti_lr3_lr4_bs512_sprit1'

feature_train_file='/home/zhenzhengxin/feature_single_class/label/place_feature_train.txt'
feature_test_file='/home/zhenzhengxin/feature_single_class/label/place_feature_test.txt'
muti_train_file='/home/zhenzhengxin/feature_single_class/label/feature_train.txt'
muti_test_file='/home/zhenzhengxin/feature_single_class/label/feature_test.txt'
action_train_file='/home/zhenzhengxin/feature_single_class/label/action_train.txt'
action_test_file='/home/zhenzhengxin/feature_single_class/label/action_test.txt'
