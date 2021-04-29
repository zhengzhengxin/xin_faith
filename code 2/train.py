import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from sklearn.metrics import average_precision_score
def get_ap(groud_truth,predict):
    gts,preds = [],[]
    for gt_raw in groud_truth:
        #追加元素
        gts.extend(gt_raw.tolist())
    for pred_raw in predict:
        preds.extend(pred_raw.tolist())
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))

def train(cfg, model, train_loader, optimizer, scheduler, epoch, criterion1,criterion2):
    train_loss=0
    model.train()
    for idx,(feature,target) in enumerate(train_loader):
        feature=feature.cuda()
        target=target.view(-1).cuda()
        optimizer.zero_grad()
        output=model(feature)
        embedding = model.cosresult.cpu()
        #output=output.view(-1,2)
        loss=criterion1(output,target)+criterion2(embedding,target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, int(idx * len(feature)), len(train_loader.dataset), loss.item()))
    #学习率的一个优化，不一定需要使用
    scheduler.step()
        
        
def test(cfg, model, test_loader, criterion1, criterion2,mode='test'):
    model.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    with torch.no_grad():
        for idx,(feature,target) in enumerate(test_loader):
            feature=feature.cuda()
            target=target.view(-1).cuda()
            output=model(feature)
            embedding = model.cosresult.cpu()
            loss=criterion1(output,target)+criterion2(embedding,target)
            test_loss+=loss.item()
            #count ap
            output = F.softmax(output, dim=1)
            prob = output[:, 1]
            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            gts_raw.append(target.cpu().numpy())
            prob_raw.append(prob.cpu().numpy())
        ap = get_ap(gts_raw, prob_raw)
        return test_loss,ap,correct1

