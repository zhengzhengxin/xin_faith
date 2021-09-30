from data_loader import MutiDateset
from vit_place import *
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
    model1 = Dense_fenlei(num_classes=2,dim = 2048,dropout = 0.5).cuda()
    model2 = ViT(cfg=cfg,feature_seq=16,num_classes=1,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1,batch_normalization=False).cuda()
    model3 = ViT(cfg=cfg,feature_seq=16,num_classes=1,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1).cuda()
    optimizer1 = optim.__dict__[cfg.optim.name](model1.parameters(), **cfg.optim.setting)
    optimizer2 = optim.__dict__[cfg.optim1.name](model2.parameters(), **cfg.optim1.setting)
    optimizer3 = optim.__dict__[cfg.optim2.name](model3.parameters(), **cfg.optim2.setting)
    #在指定的epoch对其进行衰减
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)

    criterion3 = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    #criterion1 = nn.BCEWithLogitsLoss()
    criterion1 = FocalLoss(logits=True)
    
    #加入对数损失
    distance = CosineSimilarity()
    criterion2 = losses.TripletMarginLoss(distance = distance)


    total_loss, total_loss_place, total_loss_tea=list(), list(), list()
    total_epoch=list()
    total_ap, total_ap_place, total_ap_tea=list(),list(),list()
    total_acc=list()
    max_ap=0
    


    for epoch in range(0,cfg.epoch):
        train_mult(cfg, model1,model2,model3, train_loader, optimizer1,optimizer2, optimizer3, scheduler, epoch, criterion1,criterion2,criterion3)
        loss,loss_place,loss_tea,ap,ap_place,ap_tea,acc=test_mult(cfg, model1, model2, model3, test_loader, criterion1,criterion2,criterion3)
        total_loss.append(loss)
        total_ap.append(ap)
        total_loss_place.append(loss_place)
        total_ap_place.append(ap_place)
        total_loss_tea.append(loss_tea)
        total_ap_tea.append(ap_tea)
        total_epoch.append(epoch)
        print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\tacc: {:.6f}'.format(epoch, loss,ap,acc))
        if ap>max_ap:
            best_model1=model1
            best_model2=model2
            best_model3=model3
            max_ap = ap
    print("the best ap:{}".format(max_ap))
    save_path1=cfg.store+'1.pth'
    torch.save(best_model1.state_dict(), save_path1)
    save_path2=cfg.store+'2.pth'
    torch.save(best_model2.state_dict(), save_path2)
    save_path3=cfg.store+'3.pth'
    torch.save(best_model3.state_dict(), save_path3)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss_place,'y^',label=u'loss_place')
    plt.plot(total_epoch,total_loss_tea,'r^',label=u'loss_tea')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path_all=cfg.store+"_loss_all.png"
    plt.savefig(loss_path_all)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.plot(total_epoch,total_ap_place,'y^',label=u'AP_place')
    plt.plot(total_epoch,total_ap_tea,'r^',label=u'AP_tea')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)

def main_conv():
    muti_train_file = cfg.muti_train_file
    muti_test_file = cfg.muti_test_file
    train_dataset = MutiDateset(muti_train_file)
    test_dataset = MutiDateset(muti_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False)
    model1 = Dense_conv(num_classes=2,dim = 2048,dropout = 0.5).cuda()
    model2 = ViT(cfg=cfg,feature_seq=16,num_classes=1,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1,batch_normalization=False).cuda()
    model3 = ViT(cfg=cfg,feature_seq=16,num_classes=1,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.1,emb_dropout = 0.1).cuda()
    optimizer1 = optim.__dict__[cfg.optim.name](model1.parameters(), **cfg.optim.setting)
    optimizer2 = optim.__dict__[cfg.optim1.name](model2.parameters(), **cfg.optim1.setting)
    optimizer3 = optim.__dict__[cfg.optim2.name](model3.parameters(), **cfg.optim2.setting)
    
    
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)

    criterion1 = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
     
    distance = CosineSimilarity()
    criterion2 = losses.TripletMarginLoss(distance = distance)


    total_loss=list()
    train_total_loss=list()
    total_epoch=list()
    total_ap, total_ap_place, total_ap_tea=list(),list(),list()
    total_acc=list()
    max_ap=0
    


    for epoch in range(0,cfg.epoch):
        train_loss = train_mult_conv(cfg, model1,model2,model3, train_loader, optimizer1,optimizer2, optimizer3, scheduler, epoch, criterion1,criterion2)
        loss,ap,acc=test_mult_conv(cfg, model1, model2, model3, test_loader, criterion1,criterion2)
        total_loss.append(loss)
        total_ap.append(ap)
        total_epoch.append(epoch)
        train_total_loss.append(train_loss)
        print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\tacc: {:.6f}'.format(epoch, loss,ap,acc))
        if ap>max_ap:
            best_model1=model1
            best_model2=model2
            best_model3=model3
            max_ap = ap
    print("the best ap:{}".format(max_ap))
    save_path1=cfg.store+'1.pth'
    torch.save(best_model1.state_dict(), save_path1)
    save_path2=cfg.store+'2.pth'
    torch.save(best_model2.state_dict(), save_path2)
    save_path3=cfg.store+'3.pth'
    torch.save(best_model3.state_dict(), save_path3)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,train_total_loss,'y^',label=u'loss_place')
    plt.legend()
    loss_path=cfg.store+"_loss_train.png"
    plt.savefig(loss_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path_all=cfg.store+"_loss_all.png"
    plt.savefig(loss_path_all)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
if __name__ == '__main__':
    main_conv()
