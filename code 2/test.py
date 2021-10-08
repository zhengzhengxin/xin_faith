import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from sklearn.metrics import average_precision_score
def get_ap(groud_truth,predict):
    gts,preds = [],[]
    for gt_raw in groud_truth:
        gts.extend(gt_raw.tolist())
    for pred_raw in predict:
        preds.extend(pred_raw.tolist())
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))
#i3d,tea,vggish渐进式训练
def test_i3d_tea_vgg(cfg, model1, model2,model_dense,test_loader, criterion,state,mode='test'):
    model1.eval()
    model2.eval()
    model_dense.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    if state ==1:
        with torch.no_grad():
            for idx,(feat_i3d,feat_tea,feat_vgg,length,target) in enumerate(test_loader):
                feat_i3d = feat_i3d.cuda()
                feat_tea = feat_tea.cuda()
                feat_vgg = feat_vgg.cuda()
                target = target.view(-1).cuda()
                _,output1 = model1(feat_i3d,length)
                output = model_dense(output1,feat_vgg)
                #output = model_dense(output)
                output = output.squeeze(dim=-1)
                #embedding = model.cosresult
                #loss = criterion2(embedding,target.float())

                loss = criterion(output,target)
                test_loss += loss.item()
                output = F.softmax(output, dim=1)
                # prob = torch.sigmoid(output)
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
            for idx,(feat_i3d,feat_tea,feat_vgg,length,target) in enumerate(test_loader):
                feat_i3d = feat_i3d.cuda()
                feat_tea = feat_tea.cuda()
                feat_vgg = feat_vgg.cuda()
                target = target.view(-1).cuda()
                _,output2 = model2(feat_i3d,length)
                output1 = model1(feat_tea)
                output = model_dense(output1,output2,feat_vgg)
                output = output.squeeze(dim=-1)
                #embedding = model.cosresult
                #loss = criterion2(embedding,target.float())

                loss = criterion(output,target)
                test_loss += loss.item()
                output = F.softmax(output, dim=1)

                # prob = torch.sigmoid(output)

                prob = output[:, 1]
                gt = target.cpu().detach().numpy()
                prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
                idx1 = np.where(gt == 1)[0]
                correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
                gts_raw.append(target.cpu().numpy())
                prob_raw.append(prob.cpu().numpy())
            ap = get_ap(gts_raw, prob_raw)
            return test_loss,ap,correct1

#i3d,tea,vggish渐进式训练，加多任务
def test_i3d_tea_vgg_emo(cfg, model1, model2,model_dense,test_loader, criterion_vio,criterion_emo,state,mode='test'):
    model1.eval()
    model2.eval()
    model_dense.eval()
    test_loss=0
    prob_raw, gts_raw = [], []
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    if state ==1:
        with torch.no_grad():
            for idx,(feat_i3d,feat_tea,feat_vgg,length,target_vio,target_emo) in enumerate(test_loader):
                feat_i3d = feat_i3d.cuda()
                feat_tea = feat_tea.cuda()
                feat_vgg = feat_vgg.cuda()
                target_vio = target_vio.view(-1).cuda()
                target_emo = target_emo.view(-1).cuda()
                _,output1 = model1(feat_i3d,length)
                output_vio,output_emo = model_dense(output1,feat_vgg)
                output_vio = output_vio.squeeze(dim=-1)
                output_emo = output_emo.squeeze(dim=-1)
                loss_vio=criterion_vio(output_vio,target_vio)
                loss_emo=criterion_emo(output_emo,target_emo)
                loss = 0.9*loss_vio + loss_emo
                test_loss += loss.item()
                output_vio = F.softmax(output_vio, dim=1)
                # prob = torch.sigmoid(output)
                prob = output_vio[:, 1]
                gt = target_vio.cpu().detach().numpy()
                prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
                idx1 = np.where(gt == 1)[0]
                correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
                gts_raw.append(target_vio.cpu().numpy())
                prob_raw.append(prob.cpu().numpy())
            ap = get_ap(gts_raw, prob_raw)
            return test_loss,ap,correct1
    if state ==2:
        with torch.no_grad():
            for idx,(feat_i3d,feat_tea,feat_vgg,length,target_vio,target_emo) in enumerate(test_loader):
                feat_i3d = feat_i3d.cuda()
                feat_tea = feat_tea.cuda()
                feat_vgg = feat_vgg.cuda()
                target_vio = target_vio.view(-1).cuda()
                target_emo = target_emo.view(-1).cuda()
                _,output2 = model2(feat_i3d,length)
                output1 = model1(feat_tea)
                output_vio,output_emo = model_dense(output1,output2,feat_vgg)
                output_vio = output_vio.squeeze(dim=-1)
                output_emo = output_emo.squeeze(dim=-1)
                loss_vio=criterion_vio(output_vio,target_vio)
                loss_emo=criterion_emo(output_emo,target_emo)
                loss = 0.9*loss_vio + loss_emo
                test_loss += loss.item()
                output_vio = F.softmax(output_vio, dim=1)
                prob = output_vio[:, 1]
                gt = target_vio.cpu().detach().numpy()
                prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
                idx1 = np.where(gt == 1)[0]
                correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
                gts_raw.append(target_vio.cpu().numpy())
                prob_raw.append(prob.cpu().numpy())
            ap = get_ap(gts_raw, prob_raw)
            return test_loss,ap,correct1