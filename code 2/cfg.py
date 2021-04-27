seq_len=16
lstm_hidden=1024
num_layer=1
input_dim=2048
lstm_dropout=0.8
bidirectional=True
epoch = 100
optim = dict(name='Adam',setting=dict(lr=1e-2, weight_decay=5e-4))
batch_size=16
stepper = dict(name='MultiStepLR',setting=dict(milestones=[30,50,70],gamma=0.5))

loss = dict(weight=[0.5, 5])

store='./place_result_seq16_hidd1024_drop0.8'

feature_train_file='/home/zhenzhengxin/feature_single_class/label/place_feature_train.txt'
feature_test_file='/home/zhenzhengxin/feature_single_class/label/place_feature_test.txt'
