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
        feature = feature.cuda()
        target = target.view(-1).cuda()
        optimizer.zero_grad()
        output = model(feature)
        output = output.squeeze(dim=-1)
        embedding = model.cosresult
        #output=output.view(-1,2)
        loss = criterion2(embedding,target.float())
        #loss=criterion1(output,target.float())
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
            feature = feature.cuda()
            target = target.view(-1).cuda()
            output = model(feature)
            output = output.squeeze(dim=-1)
            embedding = model.cosresult
            loss = criterion2(embedding,target.float())
            test_loss += loss.item()
            # count ap
            # output = F.softmax(output, dim=1)
            prob = torch.sigmoid(output)
            # prob = output[:, 1]
            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            gts_raw.append(target.cpu().numpy())
            prob_raw.append(prob.cpu().numpy())
        ap = get_ap(gts_raw, prob_raw)
        return test_loss,ap,correct1

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train_mult(cfg, model,train_loader, optimizer, scheduler, epoch, criterion1,criterion2,criterion3):
    train_loss=0
    model.train()
    for idx,(feat_place,feat_tea,target) in enumerate(train_loader):
        feat_place = feat_place.cuda()
        feat_tea = feat_tea.cuda()
        target = target.view(-1).cuda()
        optimizer.zero_grad()
        out_place, out_tea, emb_place, emb_tea, out = model(feat_place,feat_tea)
        out_place = out_place.squeeze(dim=-1)
        out_tea = out_tea.squeeze(dim=-1)
        out=out.view(-1,2)
        loss_place = 0.2*criterion1(out_place,target.float()) + 0.8*criterion2(emb_place,target.float())
        loss_tea = 0.9*criterion1(out_tea,target.float()) + 0.1*criterion2(emb_tea,target.float())
        loss=criterion3(out,target) + loss_place + loss_tea
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, int(idx * len(feat_place)), len(train_loader.dataset), loss.item()))
    #学习率的一个优化，不一定需要使用
    scheduler.step()

def test_mult(cfg, model, test_loader, criterion1, criterion2,criterion3,mode='test'):
    model.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    with torch.no_grad():
        for idx,(feat_place,feat_tea,target) in enumerate(test_loader):
            feat_place = feat_place.cuda()
            feat_tea = feat_tea.cuda()
            target = target.view(-1).cuda()
            out_place, out_tea, emb_place, emb_tea, out = model(feat_place,feat_tea)
            # out_place = out_place.squeeze(dim=-1)
            # out_tea = out_tea.squeeze(dim=-1)
            out=out.view(-1,2)
            # loss_place = 0.2*criterion1(out_place,target.float()) + 0.8*criterion2(emb_place,target.float())
            # loss_tea = 0.9*criterion1(out_tea,target.float()) + 0.1*criterion2(emb_tea,target.float())
            # loss=criterion1(out,target.float()) + loss_place + loss_tea
            loss = criterion3(out,target)
            test_loss += loss.item()
            # count ap
            out = F.softmax(out, dim=1)
            # prob = torch.sigmoid(output)
            prob = out[:, 1]
            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            gts_raw.append(target.cpu().numpy())
            prob_raw.append(prob.cpu().numpy())
        ap = get_ap(gts_raw, prob_raw)
        return test_loss,ap,correct1