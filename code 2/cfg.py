seq_len=16
lstm_hidden=1024
num_layer=1
input_dim=2048
lstm_dropout=0.5
bidirectional=True
epoch = 2800
optim = dict(name='SGD',setting=dict(lr=1e-2, weight_decay=5e-3))
optim1 = dict(name='SGD',setting=dict(lr=1e-2, weight_decay=5e-3))
optim2 = dict(name='SGD',setting=dict(lr=5e-3, weight_decay=5e-3))
batch_size=256

stepper = dict(name='MultiStepLR',setting=dict(milestones=[],gamma=0.5))

loss = dict(weight=[1.0, 20.0])


store='./ali/i3d12_tea53_vgg_bs256_wei_fc21'


feature_train_file='/home/zhenzhengxin/feature_single_class/label/place_feature_train.txt'
feature_test_file='/home/zhenzhengxin/feature_single_class/label/place_feature_test.txt'
muti_train_file='/home/zhenzhengxin/feature_single_class/label/feature_train.txt'
muti_test_file='/home/zhenzhengxin/feature_single_class/label/feature_test.txt'
i3d_tea_train_file='/home/zhenzhengxin/feature_single_class/label/i3d_tea_train.txt'
i3d_tea_test_file='/home/zhenzhengxin/feature_single_class/label/i3d_tea_test.txt'
action_train_file='/home/zhenzhengxin/feature_single_class/label/action_train.txt'
action_test_file='/home/zhenzhengxin/feature_single_class/label/action_test.txt'
i3d_tea_vggish_train_file = '/home/zhenzhengxin/feature_single_class/label/i3d_tea_vggish_train.txt'
i3d_tea_vggish_test_file = '/home/zhenzhengxin/feature_single_class/label/i3d_tea_vggish_test.txt'
