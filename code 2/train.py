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
        
def train_action(cfg, model, train_loader, optimizer, scheduler, epoch, criterion1):
    train_loss=0
    model.train()
    for idx,(feature,length,target) in enumerate(train_loader):
        feature = feature.cuda()
        target = target.view(-1).cuda()
        optimizer.zero_grad()
        output = model(feature,length)
        #???1?????
        output = output.squeeze(dim=-1)
        #embedding = model.cosresult
        #output=output.view(-1,2)
        #loss = criterion2(embedding,target.float())
        loss=criterion1(output,target)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, int(idx * len(feature)), len(train_loader.dataset), loss.item()))
    #学习率的一个优化，不一定需要使用
    scheduler.step()
def test_action(cfg, model, test_loader, criterion1,mode='test'):
    model.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    with torch.no_grad():
        for idx,(feature,length,target) in enumerate(test_loader):
            feature = feature.cuda()
            target = target.view(-1).cuda()
            output = model(feature,length)
            output = output.squeeze(dim=-1)
            #embedding = model.cosresult
            #loss = criterion2(embedding,target.float())
            loss=criterion1(output,target)
            test_loss += loss.item()
            output = F.softmax(output, dim=1)
            prob = output[:, 1]
            #prob = torch.sigmoid(output)
            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            gts_raw.append(target.cpu().numpy())
            prob_raw.append(prob.cpu().numpy())
        ap = get_ap(gts_raw, prob_raw)
        return test_loss,ap,correct1     
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

def train_mult(cfg, model1,model2,model3,train_loader, optimizer1, optimizer2,optimizer3,scheduler, epoch, criterion1,criterion2,criterion3):
    train_loss=0
    model1.train()
    model2.train()
    model3.train()
    for idx,(feat_place,feat_tea,target) in enumerate(train_loader):
        feat_place = feat_place.cuda()
        feat_tea = feat_tea.cuda()
        target = target.view(-1).cuda()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        if epoch % 2 == 0:
            out_place, emb_place= model2(feat_place)
            out_tea, emb_tea = model3(feat_tea)
            out_place = out_place.squeeze(dim=-1)
            out_tea = out_tea.squeeze(dim=-1)

            #loss_place = 0.5*criterion1(out_place,target.float()) + 0.5*criterion2(emb_place,target.float())
            #loss_place = criterion1(out_place,target.float())
            loss_place = criterion2(emb_place,target.float())
            loss_tea = 0.5*criterion1(out_tea,target.float()) + 0.5*criterion2(emb_tea,target.float())
            loss_place.backward(retain_graph=True)
            loss_tea.backward()
            optimizer2.step()
            optimizer3.step()
        else:
            _, emb_place= model2(feat_place)
            _, emb_tea = model3(feat_tea)
            out = model1(emb_place,emb_tea)
            out=out.view(-1,2)
            loss = criterion3(out,target)
            loss.backward()
            optimizer1.step()
        # train_loss+=loss.item()
        # print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
        #         epoch, int(idx * len(feat_place)), len(train_loader.dataset), loss.item()))
    #学习率的一个优化，不一定需要使用
    #scheduler.step()

def test_mult(cfg, model1, model2,model3,test_loader, criterion1, criterion2,criterion3,mode='test'):
    model1.eval()
    model2.eval()
    model3.eval()
    test_loss,test_loss_place,test_loss_tea=0, 0, 0
    prob_raw, gts_raw, prob_raw_place, prob_raw_tea = [], [], [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    with torch.no_grad():
        for idx,(feat_place,feat_tea,target) in enumerate(test_loader):
            feat_place = feat_place.cuda()
            feat_tea = feat_tea.cuda()
            target = target.view(-1).cuda()
            out_place, emb_place= model2(feat_place)
            out_tea, emb_tea = model3(feat_tea)
            out = model1(emb_place,emb_tea)
            out_place = out_place.squeeze(dim=-1)
            out_tea = out_tea.squeeze(dim=-1)
            out=out.view(-1,2)

            #loss_place = 0.5*criterion1(out_place,target.float()) + 0.5*criterion2(emb_place,target.float())
            #loss_place = criterion1(out_place,target.float())
            loss_place = criterion2(emb_place,target.float())
            loss_tea = 0.5*criterion1(out_tea,target.float()) + 0.5*criterion2(emb_tea,target.float())
            #loss=criterion1(out,target.float()) + loss_place + loss_tea
            loss = criterion3(out,target)
            test_loss += loss.item()
            test_loss_place += loss_place.item()
            test_loss_tea += loss_tea.item()
            # count ap
            out = F.softmax(out, dim=1)
            prob_tea = torch.sigmoid(out_tea)
            prob_place = torch.sigmoid(out_place)
            prob = out[:, 1]
            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            gts_raw.append(target.cpu().numpy())
            prob_raw.append(prob.cpu().numpy())
            prob_raw_tea.append(prob_tea.cpu().numpy())
            prob_raw_place.append(prob_place.cpu().numpy())
        ap = get_ap(gts_raw, prob_raw)
        ap_tea = get_ap(gts_raw, prob_raw_tea)
        ap_place = get_ap(gts_raw, prob_raw_place)
        return test_loss,test_loss_place,test_loss_tea,ap,ap_place,ap_tea,correct1
        
#?????????
def train_two(cfg, model1, model2,model_dense,train_loader, optimizer1,optimizer_dense, scheduler1,scheduler_dense, epoch, criterion,state):
    if state == 1:
        train_loss=0
        model1.train()
        model2.train()
        model_dense.train()
        #?????????
        for idx,(feat_place,feat_tea,target) in enumerate(train_loader):
            feat_place = feat_place.cuda()
            feat_tea = feat_tea.cuda()
            target = target.view(-1).cuda()
            optimizer1.zero_grad()
            optimizer_dense.zero_grad()
            output = model1(feat_tea)
            output = model_dense(output)
            output = output.squeeze(dim=-1)
            #output=output.view(-1,2)
            #loss = criterion2(embedding,target.float())
            loss=criterion(output,target)
            loss.backward()
            optimizer1.step()
            optimizer_dense.step()
            train_loss+=loss.item()
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, int(idx * len(feat_tea)), len(train_loader.dataset), loss.item()))
    if state == 2:
        train_loss=0
        model1.train()
        model2.train()
        for k,v in model2.named_parameters():
          v.requires_grad=False#????
        model_dense.train()
        for idx,(feat_place,feat_tea,target) in enumerate(train_loader):
            feat_place = feat_place.cuda()
            feat_tea = feat_tea.cuda()
            target = target.view(-1).cuda()
            optimizer1.zero_grad()
            optimizer_dense.zero_grad()
            output2 = model2(feat_tea)
            output1 = model1(feat_place)
            output = torch.cat((output1,output2),1)
            output = model_dense(output)
            output = output.squeeze(dim=-1)
            loss=criterion(output,target)
            loss.backward()
            optimizer1.step()
            optimizer_dense.step()
            train_loss+=loss.item()
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, int(idx * len(feat_tea)), len(train_loader.dataset), loss.item()))

def test_two(cfg, model1, model2,model_dense,test_loader, criterion,state,mode='test'):
    model1.eval()
    model2.eval()
    model_dense.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    if state ==1:
        with torch.no_grad():
            for idx,(feat_place,feat_tea,target) in enumerate(test_loader):
                feat_place = feat_place.cuda()
                feat_tea = feat_tea.cuda()
                target = target.view(-1).cuda()
                output = model1(feat_tea)
                output = model_dense(output)
                output = output.squeeze(dim=-1)
                #embedding = model.cosresult
                #loss = criterion2(embedding,target.float())

                #cross entpy ?target???,????float
                loss = criterion(output,target)
                test_loss += loss.item()
                #??????????
                output = F.softmax(output, dim=1)

                # prob = torch.sigmoid(output)

                # ????1????
                prob = output[:, 1]
                gt = target.cpu().detach().numpy()
                prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
                idx1 = np.where(gt == 1)[0]
                correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
                gts_raw.append(target.cpu().numpy())
                prob_raw.append(prob.cpu().numpy())
            ap = get_ap(gts_raw, prob_raw)
            return test_loss,ap,correct1
    if state ==2:
        with torch.no_grad():
            for idx,(feat_place,feat_tea,target) in enumerate(test_loader):
                feat_place = feat_place.cuda()
                feat_tea = feat_tea.cuda()
                target = target.view(-1).cuda()
                output1 = model1(feat_place)
                output2 = model2(feat_tea)
                output = torch.cat((output1,output2),1)
                output = model_dense(output)
                output = output.squeeze(dim=-1)
                #embedding = model.cosresult
                #loss = criterion2(embedding,target.float())

                #cross entpy ?target???,????float
                loss = criterion(output,target)
                test_loss += loss.item()
                output = F.softmax(output, dim=1)

                # prob = torch.sigmoid(output)

                # ????1????
                prob = output[:, 1]
                gt = target.cpu().detach().numpy()
                prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
                idx1 = np.where(gt == 1)[0]
                correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
                gts_raw.append(target.cpu().numpy())
                prob_raw.append(prob.cpu().numpy())
            ap = get_ap(gts_raw, prob_raw)
            return test_loss,ap,correct1
