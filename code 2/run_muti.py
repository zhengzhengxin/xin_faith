from data_loader import MutiDateset
from vit_place import Dense_fenlei
from vit_place_cat import ViT_cat
from torch.utils.data import DataLoader
from train import *
from mmcv import Config
import argparse 
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)


def main():
    muti_train_file = cfg.muti_train_file
    muti_test_file = cfg.muti_test_file
    train_dataset = MutiDateset(muti_train_file)
    test_dataset = MutiDateset(muti_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False)
    model = Dense_fenlei(cfg=cfg,feature_seq=16,num_classes=2,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1).cuda()
    optimizer1 = optim.__dict__[cfg.optim1.name](filter(lambda p: p.requires_grad, model.parameters()), **cfg.optim1.setting)
    optimizer2 = optim.__dict__[cfg.optim2.name](filter(lambda p: p.requires_grad, model.parameters()), **cfg.optim2.setting)

    #在指定的epoch对其进行衰减
    scheduler1 = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)
    scheduler2 = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)
    criterion3 = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    #criterion1 = nn.BCEWithLogitsLoss()
    criterion1 = FocalLoss(logits=True)
    
    #加入对数损失
    distance = CosineSimilarity()
    criterion2 = losses.TripletMarginLoss(distance = distance)


    total_loss, total_tea_loss, total_place_loss=list(), list(), list()
    total_epoch=list()
    total_ap, total_tea_ap, total_place_ap=list(), list(), list()
    total_acc=list()
    max_ap=0

    for epoch in range(0,cfg.epoch):
        train_mult(cfg, model, train_loader, optimizer1,optimizer2, scheduler1,epoch, criterion1,criterion2,criterion3)
        loss,tea_loss,place_loss,ap,acc,ap_tea,ap_place=test_mult(cfg, model, test_loader, criterion1,criterion2,criterion3)
        total_loss.append(loss)
        total_tea_loss.append(tea_loss)
        total_place_loss.append(place_loss)
        total_ap.append(ap)
        total_tea_ap.append(ap_tea)
        total_place_ap.append(ap_place)
        total_epoch.append(epoch)
        total_acc.append(acc)
        print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\tacc: {:.6f}'.format(epoch, loss,ap,acc))
        if ap>max_ap:
            best_model=model
    save_path=cfg.store+'.pth'
    torch.save(best_model.state_dict(), save_path)
    pdb.set_trace()
    plt.figure(figsize=(30,20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.plot(total_epoch,total_tea_loss,'r^',label=u'tea_loss')
    plt.plot(total_epoch,total_place_loss,'y^',label=u'place_loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)
    
    plt.figure(figsize=(30,20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.plot(total_epoch,total_tea_ap,'r^',label=u'tea_AP')
    plt.plot(total_epoch,total_place_ap,'y^',label=u'place_AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure()
    plt.plot(total_epoch,total_acc,'b-',label=u'acc')
    plt.legend()
    acc_path=cfg.store+"_acc.png"
    plt.savefig(acc_path)



if __name__ == '__main__':
    main()
