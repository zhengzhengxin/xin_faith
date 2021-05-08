seq_len=16
lstm_hidden=1024
num_layer=1
input_dim=2048
lstm_dropout=0.5
bidirectional=True
epoch = 100
optim1 = dict(name='Adam',setting=dict(lr=1e-3, weight_decay=5e-3))
optim2 = dict(name='Adam',setting=dict(lr=1e-3, weight_decay=5e-3))
optim3 = dict(name='Adam',setting=dict(lr=1e-3, weight_decay=5e-3))
batch_size=128
stepper = dict(name='MultiStepLR',setting=dict(milestones=[],gamma=0.5))

loss = dict(weight=[0.5, 11.0])

store='./muti_lr3_bs128_wei'

feature_train_file='/home/zhenzhengxin/feature_single_class/label/place_feature_train.txt'
feature_test_file='/home/zhenzhengxin/feature_single_class/label/place_feature_test.txt'
muti_train_file='/home/zhenzhengxin/feature_single_class/label/feature_train.txt'
muti_test_file='/home/zhenzhengxin/feature_single_class/label/feature_test.txt'
