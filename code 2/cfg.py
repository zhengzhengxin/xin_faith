seq_len=16
lstm_hidden=1024
num_layer=1
input_dim=2048
lstm_dropout=0.5
bidirectional=True
epoch = 100
optim = dict(name='Adam',setting=dict(lr=1e-5, weight_decay=5e-4))
batch_size=512
stepper = dict(name='MultiStepLR',setting=dict(milestones=[],gamma=0.1))

loss = dict(weight=[1.0, 1.0])

store='./tea_transformer_bs512_lr5_bce_tri_bn'

feature_train_file='/home/zhenzhengxin/feature_single_class/label/tea_feature_train.txt'
feature_test_file='/home/zhenzhengxin/feature_single_class/label/tea_feature_test.txt'
