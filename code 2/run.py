from data_loader import *
from model import *
from torch.utils.data import DataLoader
from train import *
from mmcv import Config
import argparse 
import torch.optim as optim
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
import pdb
def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config.fromfile(args.config)


def main():
    feature_train_file=cfg.feature_train_file
    feature_test_file=cfg.feature_test_file
    train_dataset=PlaceDateset(feature_train_file)
    test_dataset= PlaceDateset(feature_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False)
    model=Feature_class(cfg).cuda()
    optimizer = optim.__dict__[cfg.optim.name](model.parameters(), **cfg.optim.setting)
    #在指定的epoch对其进行衰减
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer, **cfg.stepper.setting)

    criterion1 = nn.CrossEntropyLoss()
    distance = CosineSimilarity()
    criterion2 = losses.TripletMarginLoss(distance = distance)

    total_loss=list()
    total_epoch=list()
    total_ap=list()
    total_acc=list()
    max_ap=0
    for epoch in range(0,cfg.epoch):
        train(cfg, model, train_loader, optimizer, scheduler, epoch, criterion1,criterion2)
        loss,ap,acc=test(cfg, model, test_loader, criterion1, criterion2)
        total_loss.append(loss)
        total_ap.append(ap)
        total_epoch.append(epoch)
        total_acc.append(acc)
        print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
        if ap>max_ap:
            best_model=model
    save_path=cfg.store+'.pth'
    torch.save(best_model.state_dict(), save_path)
    
    plt.figure()
    plt.plot(total_epoch,total_loss,'b-',label=u'loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)
    
    plt.figure()
    plt.plot(total_epoch,total_ap,'b-',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure()
    plt.plot(total_epoch,total_acc,'b-',label=u'acc')
    plt.legend()
    acc_path=cfg.store+"_acc.png"
    plt.savefig(acc_path)

def main_action():
    feature_train_file=cfg.action_train_file
    feature_test_file=cfg.action_test_file
    train_dataset=actionDataset(feature_train_file)
    test_dataset= actionDataset(feature_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False,collate_fn=collate_fn)

    model = Action_class(cfg).cuda()
    optimizer = optim.__dict__[cfg.optim.name](model.parameters(), **cfg.optim.setting)
    criterion1 = FocalLoss(logits=True)
    criterion2 = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer, **cfg.stepper.setting)
    total_loss=list()
    total_epoch=list()
    total_ap=list()
    total_acc=list()
    max_ap=0
    for epoch in range(0,cfg.epoch):
        train_action(cfg, model, train_loader, optimizer, scheduler, epoch, criterion2)
        loss,ap,acc=test_action(cfg, model, test_loader, criterion2)
        total_loss.append(loss)
        total_ap.append(ap)
        total_epoch.append(epoch)
        total_acc.append(acc)
        print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
        if ap>max_ap:
            max_ap = ap
            best_model=model
    print("the best ap:{}".format(max_ap))
    
    save_path=cfg.store+'.pth'
    torch.save(best_model.state_dict(), save_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)

def main_fusion():
    muti_train_file = cfg.muti_train_file
    muti_test_file = cfg.muti_test_file
    train_dataset = MutiDateset(muti_train_file)
    test_dataset = MutiDateset(muti_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False)
    model1 = fusion_feat(cfg=cfg,feature_seq=16,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.5,emb_dropout = 0.5).cuda()
    model2 = fusion_feat(cfg=cfg,feature_seq=16,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.5,emb_dropout = 0.5).cuda()
    model_dense = fusion(dim = 2048,num_classes=2).cuda()
    optimizer1 = optim.__dict__[cfg.optim1.name](model1.parameters(), **cfg.optim.setting)
    optimizer2 = optim.__dict__[cfg.optim2.name](model2.parameters(), **cfg.optim.setting)
    optimizer_dense = optim.__dict__[cfg.optim.name](model_dense.parameters(), **cfg.optim.setting)
    #????epoch??????
    scheduler1 = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)
    scheduler_dense = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer_dense, **cfg.stepper.setting)

    #CrossEntropyLoss()?target???????,??one-hot????
    criterion = nn.CrossEntropyLoss()
    #distance = CosineSimilarity()
    #criterion2 = losses.TripletMarginLoss(distance = distance)

    total_loss=list()
    total_epoch=list()
    total_ap=list()
    total_acc=list()
    max_ap=0
    for epoch in range(0,cfg.epoch):
        if epoch<1000:
            state = 1;
            train_two(cfg, model1, model2,model_dense,train_loader, optimizer1,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            #???model2???
            loss,ap,acc=test_two(cfg, model1,model2, model_dense,test_loader,criterion,state)
            total_loss.append(loss)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap:
                best_model1=model1
        if epoch>1000:
            state = 2;
            model1 = best_model1
            model_dense = fusion(dim = 4096 ,num_classes=2).cuda()
                #model2?opti2??????
            train_two(cfg, model2, model1,model_dense,train_loader, optimizer2,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            #?????model2,???1
            loss,ap,acc=test_two(cfg, model2, model1,model_dense,test_loader,criterion,state)
            total_loss.append(loss)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap:
                best_model2=model2
                best_dense = model_dense
    save_path1=cfg.store+'model1.pth'
    save_path2=cfg.store+'model2.pth'
    save_path3=cfg.store+'model_dense.pth'
    torch.save(best_model1.state_dict(), save_path1)
    torch.save(best_model2.state_dict(), save_path2)
    torch.save(best_dense.state_dict(), save_path3)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)


def main_i3d_tea():
    muti_train_file = cfg.i3d_tea_train_file
    muti_test_file = cfg.i3d_tea_test_file
    train_dataset = I3d_tea_Dateset(muti_train_file)
    test_dataset = I3d_tea_Dateset(muti_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,collate_fn=collate_fn1)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False,collate_fn=collate_fn1)
    model1 = Action_class(cfg).cuda()
    model2 = fusion_feat(cfg=cfg,feature_seq=16,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.5,emb_dropout = 0.5).cuda()
    model_dense = fusion(dim = 2048,num_classes=2).cuda()
    optimizer1 = optim.__dict__[cfg.optim1.name](model1.parameters(), **cfg.optim.setting)
    optimizer2 = optim.__dict__[cfg.optim2.name](model2.parameters(), **cfg.optim.setting)
    optimizer_dense = optim.__dict__[cfg.optim.name](model_dense.parameters(), **cfg.optim.setting)
    #????epoch??????
    scheduler1 = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)
    scheduler_dense = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer_dense, **cfg.stepper.setting)

    #CrossEntropyLoss()?target???????,??one-hot????
    criterion = nn.CrossEntropyLoss()
    #distance = CosineSimilarity()
    #criterion2 = losses.TripletMarginLoss(distance = distance)

    total_loss=list()
    total_epoch=list()
    total_ap=list()
    total_acc=list()
    max_ap = 0
    max_ap2 = 0
    for epoch in range(0,cfg.epoch):
        if epoch<1200:
            state = 1;
            train_i3d_tea(cfg, model1, model2,model_dense,train_loader, optimizer1,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            #???model2???
            loss,ap,acc=test_i3d_tea(cfg, model1,model2, model_dense,test_loader,criterion,state)
            total_loss.append(loss)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap:
                best_model1=model1
                max_ap = ap
        if epoch>1200:
            state = 2;
            model1 = best_model1
            model_dense = fusion(dim = 4096 ,num_classes=2).cuda()
                #model2?opti2??????
            train_i3d_tea(cfg, model2, model1,model_dense,train_loader, optimizer2,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            #?????model2,???1
            loss,ap,acc=test_i3d_tea(cfg, model2, model1,model_dense,test_loader,criterion,state)
            total_loss.append(loss)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap2:
                best_model2 = model2
                best_dense = model_dense
                max_ap2 = ap
    
    save_path1=cfg.store+'model1.pth'
    save_path2=cfg.store+'model2.pth'
    save_path3=cfg.store+'model_dense.pth'
    torch.save(best_model1.state_dict(), save_path1)
    torch.save(best_model2.state_dict(), save_path2)
    torch.save(best_dense.state_dict(), save_path3)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)

def main_i3d_tea_vggish():
    muti_train_file = cfg.i3d_tea_vggish_train_file
    muti_test_file = cfg.i3d_tea_vggish_test_file
    train_dataset = I3d_tea_vgg_Dateset(muti_train_file)
    test_dataset = I3d_tea_vgg_Dateset(muti_test_file)

    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,collate_fn=collate_fn2)
    test_loader = DataLoader(test_dataset,batch_size=cfg.batch_size ,shuffle=False,collate_fn=collate_fn2)
    model1 = Action_class(cfg).cuda()
    model2 = fusion_feat(cfg=cfg,feature_seq=16,dim=2048,depth=8,heads=8,mlp_dim=1024,dropout = 0.5,emb_dropout = 0.5).cuda()
    optimizer1 = optim.__dict__[cfg.optim1.name](model1.parameters(), **cfg.optim.setting)
    optimizer2 = optim.__dict__[cfg.optim2.name](model2.parameters(), **cfg.optim.setting)
    
    scheduler1 = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer1, **cfg.stepper.setting)
    
    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    #distance = CosineSimilarity()
    #criterion2 = losses.TripletMarginLoss(distance = distance)

    total_loss=list()
    total_epoch=list()
    total_ap=list()
    total_acc=list()
    max_ap = 0
    max_ap2 = 0
    for epoch in range(0,cfg.epoch):
        if epoch<1200:
            state = 1;
            model_dense = fusion1(dim = 2048,num_classes=2).cuda()
            optimizer_dense = optim.__dict__[cfg.optim.name](model_dense.parameters(), **cfg.optim.setting)
            scheduler_dense = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer_dense, **cfg.stepper.setting)
            loss_1 = train_i3d_tea_vgg(cfg, model1, model2,model_dense,train_loader, optimizer1,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            loss,ap,acc=test_i3d_tea_vgg(cfg, model1,model2, model_dense,test_loader,criterion,state)
            total_loss.append(loss_1)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap:
                best_model1=model1
                max_ap = ap
        if epoch>1200:
            state = 2;
            model1 = best_model1
            model_dense = fusion2(dim = 2048 ,num_classes=2).cuda()
            optimizer_dense = optim.__dict__[cfg.optim.name](model_dense.parameters(), **cfg.optim.setting)
            scheduler_dense = optim.lr_scheduler.__dict__[cfg.stepper.name](optimizer_dense, **cfg.stepper.setting)
            loss_1 = train_i3d_tea_vgg(cfg, model2, model1,model_dense,train_loader, optimizer2,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state)
            loss,ap,acc=test_i3d_tea_vgg(cfg, model2, model1,model_dense,test_loader,criterion,state)
            total_loss.append(loss_1)
            total_ap.append(ap)
            total_epoch.append(epoch)
            total_acc.append(acc)
            print('Test Epoch: {} \tloss: {:.6f}\tap: {:.6f}\t acc: {:.6f}'.format(epoch, loss,ap,acc))
            if ap>max_ap2:
                best_model2 = model2
                best_dense = model_dense
                max_ap2 = ap
    print('best ap: {:.6f}\t'.format(max_ap2))
    save_path1=cfg.store+'model1.pth'
    save_path2=cfg.store+'model2.pth'
    save_path3=cfg.store+'model_dense.pth'
    torch.save(best_model1.state_dict(), save_path1)
    torch.save(best_model2.state_dict(), save_path2)
    torch.save(best_dense.state_dict(), save_path3)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_ap,'b^',label=u'AP')
    plt.legend()
    AP_path=cfg.store+"_AP.png"
    plt.savefig(AP_path)
    
    plt.figure(figsize=(20, 20))
    plt.plot(total_epoch,total_loss,'b^',label=u'loss')
    plt.legend()
    loss_path=cfg.store+"_loss.png"
    plt.savefig(loss_path)
if __name__ == '__main__':
    main_i3d_tea_vggish()
