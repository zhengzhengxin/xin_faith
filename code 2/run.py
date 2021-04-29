from data_loader import PlaceDateset
from model import Feature_class
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

if __name__ == '__main__':
    main()