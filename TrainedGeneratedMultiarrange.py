# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os

import argparse

from statistics import mean as avmean
import numpy as np
import torch
import torch.nn as nn

import math

from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings

from models import efficientunet as efficientnet

import CustomDataLoaderGenerated as DP

from torch.utils.data import DataLoader

from PIL import Image
CUDA_LAUNCH_BLOCKING=1

eps=0.00001
epoch_Start = 0  # 34
useOrg=True
useG=True
useGN=True
labels2Group=[0, 1, 1,1, 2 ,2 ,2 ,2,2]#[0, 1, 1,2, 2 ,2 ,3 ,3,3]
counteach=[0, 0, 0, 0, 0 ,0 ,0 ,0,0]




def get_intersection(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2] + bb1[0], bb2[0] + bb2[2])
    y_bottom = min(bb1[3] + bb1[1], bb2[1] + bb2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return (0, 0, 0, 0)
    return (x_left, x_right, y_top, y_bottom)


def train_sesion(n_iter,batch_index, countpred,countpredT,Total,images, labels, bbbox, numbb,best_accval,lambada,prifix=''):
    loss = None
    countar = 0
    iter = 0
    images = Variable(images)
    seglabel = (labels // 28).long().cuda()  # .unsqueeze(1)
    bbisemlist = []
    indexis = 0
    for bblis in bbbox:
        bbisemlist.append(bblis[:numbb[indexis], :].tolist())
        indexis += 1

        # bbisemlist.append(cv2.boundingRect(imgforbb.astype(np.uint8)))
    seglabel2 = (labels > 0).long().cuda()
    countAll = (labels > 0).sum(dim=1).sum(dim=1)
    seglabelMaxima, _ = torch.max(seglabel, dim=1)
    seglabelMaxima, _ = torch.max(seglabelMaxima, dim=1)
    countexx = 0
    labelsGroups = torch.zeros_like(seglabelMaxima)
    for labmax in seglabelMaxima:
        labelsGroups[countexx] = labels2Group[labmax]
        counteach[labmax] += 1
        countexx += 1
    labelsGroup = Variable(labelsGroups * (countAll > 0).cuda().long())
    if prifix=='O':
        labels = Variable((countAll > 0).cuda().long())
    else:
        labels = Variable((seglabelMaxima) * (countAll > 0).cuda().long())
    countAllv = Variable((countAll > 0).cuda().long())

    labels = labels.long().cuda()
    images = images.float().cuda()


    ##########################################################

    imagesperm = images.permute(0, 3, 1, 2)

    outputs, masklist, _ = net(imagesperm)

    if batch_index == 0:
        countpred = [0] * len(outputs[:-1])
        countpredT = [0] * len(outputs[:-1])
        Total = [0] * len(outputs[:-1])

    if 1:  # masksize is None:
        masksize = []
        for maskerrer in masklist:
            masksize.append(maskerrer.detach().shape[3] / 256)
    energyind = 0
    loser = None

    ThreshReg=0.9
    if prifix == 'O':
        ThreshReg=0.75
    if best_accval < ThreshReg:
        for maskerrer in masklist:

            indmimas = 0
            for masksel in maskerrer:
                relbblist = bbisemlist[indmimas]
                energy = 0
                factorred = 0
                if energyind >= 0 and energyind <= 2:
                    for relbb in relbblist:

                        if relbb[3] == 0:
                            if 0:
                                if loser is None:
                                    loser = torch.norm(torch.norm(masksel.squeeze()))
                                else:
                                    loser += torch.norm(torch.norm(masksel.squeeze()))
                        else:
                            relbbdec = [int(relbb[0] * masksize[energyind]), int(relbb[1] * masksize[energyind]),
                                        int(np.ceil((relbb[2] * masksize[energyind])) + 1),
                                        int(np.ceil((relbb[3] * masksize[energyind])) + 1)]

                            energy += (masksel.squeeze()[relbbdec[1]:(relbbdec[1] + relbbdec[3]),
                                       relbbdec[0]:(relbbdec[0] + relbbdec[2])] ** 2).sum()
                            factorred += (relbbdec[2] * relbbdec[3])
                    if relbblist:
                        if factorred == 256 * masksize[energyind] * 256 * masksize[energyind]:
                            factorred=0
                        else:
                            factorred = factorred / (256 * masksize[energyind] * 256 * masksize[energyind] - factorred)
                        if loser is None:
                            loserT = factorred * torch.nn.functional.relu(torch.sqrt(
                                eps + ((masksel.squeeze() ** 2).sum() - energy) / (energy + eps)) - 0.5)
                            if torch.isnan(loserT) or torch.isinf(loserT):
                                loserT = 0

                            else:
                                loser = loserT
                        else:
                            loserT = factorred * torch.nn.functional.relu(torch.sqrt(
                                eps + ((masksel.squeeze() ** 2).sum() - energy) / (energy + eps)) - 0.5)
                            if torch.isnan(loserT) or torch.isinf(loserT):
                                loserT = 0
                            else:
                                loser += loserT

                indmimas += 1
            energyind += 1

    else:
        if prifix == 'O':
            print('stop_loss_mask')
            lambada *= 0.01

        # segcheck=(seglabel2  [indmimas,relbb[1]:(relbb[1]+relbb[3]),relbb[0]:(relbb[0]+relbb[2])].squeeze()).sum()
    if loser is None:
        print("loser error")
    if loser is not None:
        writer.add_scalar('Train/loss_mask'+ prifix + str(countar), loser.item(), n_iter)


    ind = 0
    for output in outputs[:-1]:

        if iter >= len(outputs) - 3:
            lambada = lambada  # lambada*=10
        if prifix != 'O':
            lossi = 0.0001 * lambada * loss_function(output[:, :4], labelsGroup).mean()
            if math.isnan(lossi.item()):
                print("error")

        lossj = 2 * lambada * (loss_function(output[:, 10:12], countAllv)).mean()


        if math.isnan(lossj.item()):
            print("error")
        if prifix != 'O':
            _, preds = output[:, :4].max(1)
            correctLabels = preds.eq(labelsGroup)
            countpred[ind] += correctLabels.sum().item()

        _, predsT = output[:, 10:12].max(1)

        correctExist = predsT.eq(countAllv)
        Total[ind] += len(correctExist)
        countpredT[ind] += correctExist.sum().item()

        if Total[ind] >= 280:
            writer.add_scalar('Train/acc_EX' + prifix + str(ind), countpredT[ind] / Total[ind], n_iter)
            if prifix != 'O':
                writer.add_scalar('Train/acc_' + prifix + str(ind), countpred[ind] / Total[ind], n_iter)
        ind += 1
        writer.add_scalar('Train/lossit_EX' + prifix + str(countar), lossj.item(), n_iter)
        if prifix != 'O':
            writer.add_scalar('Train/lossit_' + prifix + str(countar), lossi.item(), n_iter)
        countar += 1
        if ind >= 1 and ind <= 4:
            if loss is None:
                if prifix == 'O':
                    loss = lossj
                else:
                    loss = lossi + lossj
            else:
                if prifix == 'O':
                    loss += lossj
                else:
                    loss += lossi + lossj

        iter += 1
        if math.isnan(loss.item()):
            print("error")
    out = outputs[-1]


    if prifix != 'O':
        loss += 0.01 * lambada * segloss(out[:, :10, :, :], seglabel)
    loss += 1 * lambada * segloss(out[:, 10:12, :, :], seglabel2)
    writer.add_scalar('Train/loss' +prifix , loss.item(), n_iter)
    # loss+=loser
    if loser is None:
        if math.isnan(loss.item()):
            print("Nan Error")
        else:

            loss.backward()
    else:
        if math.isnan(loss.item()) or math.isnan(loser.item()):
            print("Nan Error")
        else:
            (loss + loser).backward()
    lossval=loss.item()
    if loser is not None:
        del loser
    if loss is not None:
        del loss
    return lossval,  countpred,countpredT,Total

def train(epoch,best_accval=0,correctmax=0):
    print("best_accval")
    print(best_accval)
    print(correctmax)
    writer.add_scalar('Train/learningRate', optimizer.param_groups[0]['lr'], epoch)
    net.train()


    optimizer.zero_grad()
    masksize = None
    maxaccTotal=0.0


    selectedtrain_dataloder = train_dataloder
    selectedtrain_dataloder3 = train_dataloder3

    tditer =train_dataloder2 .__iter__()
    tditer3 = selectedtrain_dataloder3.__iter__()

    countpredO = None
    countpredTO = None
    TotalO = None
    countpred = None
    countpredT = None
    Total = None
    countpredN = None
    countpredTN = None
    TotalN = None


    for batch_index, (images, labels, bbbox, numbb) in enumerate(selectedtrain_dataloder):
        if batch_index>1000 and correctmax>0.81:
            break
        if 1:
            try:
                imagesO, labelsO, bbboxO, numbbO = tditer.__next__()
                nano=imagesO.sum()
            except StopIteration:
                tditer = train_dataloder2 .__iter__()
                imagesO, labelsO, bbboxO, numbbO = tditer.__next__()
            if numbbO.shape[0] < tditer._dataset.batch_size:
                tditer = train_dataloder2.__iter__()
                imagesO, labelsO, bbboxO, numbbO = tditer.__next__()
        if 1:
            try:
                imagesN, labelsN, bbboxN, numbbN = tditer3.__next__()
                nano = imagesO.sum()
            except StopIteration:
                tditer3 = selectedtrain_dataloder3.__iter__()
                imagesN, labelsN, bbboxN, numbbN = tditer3.__next__()
            if numbbN.shape[0] < tditer3._dataset.batch_size:
                tditer3 = selectedtrain_dataloder3.__iter__()
                imagesN, labelsN, bbboxN, numbbN = tditer3.__next__()

        n_iter = (epoch) * len(selectedtrain_dataloder) + batch_index + 1
        writer.add_scalar('Train/learningRate', optimizer.param_groups[0]['lr'], n_iter)

        lambada = 0.1



        if useOrg:


            lossO,countpredO,countpredTO,TotalO=train_sesion(n_iter,batch_index, countpredO,countpredTO,TotalO, imagesO, labelsO, bbboxO, numbbO,best_accval,lambada,prifix='O')

        lambada = 0.1

        if  useG: # הbatch_index % 10 == 0 an
            loss,countpred,countpredT,Total,=train_sesion(n_iter,batch_index, countpred,countpredT,Total, images, labels, bbbox, numbb,best_accval,lambada)

        lambada = 0.1
        if best_accval > 0.75:
            lambada*=0.01

        if  useGN:
            lossN,countpredN, countpredTN, TotalN = train_sesion(n_iter, batch_index, countpredN, countpredTN, TotalN, imagesN, labelsN, bbboxN, numbbN,
                                best_accval, lambada,prifix='N')

        if batch_index % 20 == 19:
            optimizer.step()
            optimizer.zero_grad()



        last_layer = list(net.children())[-1]
        if 0:
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)


        else:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                lossO,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(imagesO),
                total_samples=len(selectedtrain_dataloder.dataset.imagepair)
            ))

            if math.isnan(lossO) :
                print("Nan Error")
        # update training loss for each iteration
        #writer.add_scalar('Train/lossO', lossO, n_iter)

    optimizer.step()
    optimizer.zero_grad()
    if 0:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)
    print(counteach/np.sum(counteach))
    return maxaccTotal

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    meaner=iou.mean()

    return meaner
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

def calcF1_and_TPR(PositiveList,Positivesumpleslst,Negativesumpleslist,NegativeList,test,epoch):
    TP = []
    TN = []
    FP = []
    FN = []
    for n in range(len(PositiveList)):
        TP.append(PositiveList[n] / Positivesumpleslst[n])
        FN.append((Positivesumpleslst[n] - PositiveList[n]) / Positivesumpleslst[n])
        TN.append(NegativeList[n] / Negativesumpleslist[n])
        FP.append((Negativesumpleslist[n] - NegativeList[n]) / Negativesumpleslist[n])
        writer.add_scalar(test + '/AATPR' + str(n), TP[n], epoch)
        writer.add_scalar(test + '/AATNR' + str(n), TN[n], epoch)
        writer.add_scalar(test + '/AAFPR' + str(n), FP[n], epoch)
        writer.add_scalar(test + '/AAFNR' + str(n), FN[n], epoch)
        F1 = 2 * PositiveList[n].item()
        Div =  PositiveList[n].item()  + Positivesumpleslst[n] + (
                    Negativesumpleslist[n].item() - NegativeList[n].item())
        F1 = F1 / Div
        writer.add_scalar(test + '/AAF1_' + str(n), F1, epoch)


def generate_acc_satistic_per_difficulty(accuracyPerDifficulty,countPerDifficulty,test,epoch):
    for nin in range(10):
        for indcc in range(6):
            if isinstance(accuracyPerDifficulty[nin][indcc],int):
                if countPerDifficulty[nin][indcc]>0:
                    writer.add_scalar(test + '/AccuracyPer' +str(indcc)+'_'+str(nin), accuracyPerDifficulty[nin][indcc] /  countPerDifficulty[nin][indcc], epoch)
                    if nin >0 and countPerDifficulty[0][indcc] > 0:
                        writer.add_scalar(test + '/AccuracyPer' + str(indcc) + '_' + str(nin)+ 'spec',(accuracyPerDifficulty[0][indcc] / countPerDifficulty[0][indcc])+
                                          (accuracyPerDifficulty[nin][indcc] / countPerDifficulty[nin][indcc]), epoch)
                else:
                    writer.add_scalar(test + '/AccuracyPer' +str(indcc)+'_'+ str(nin), 0,epoch)
                writer.add_scalar(test + '/countcorrectPer' +str(indcc)+'_'+ str(nin), accuracyPerDifficulty[nin][indcc], epoch)
            else:
                writer.add_scalar(test + '/AccuracyPer' +str(indcc)+'_'+ str(nin),
                                  accuracyPerDifficulty[nin][indcc].item() / countPerDifficulty[nin][indcc], epoch)
                if nin > 0:
                    if countPerDifficulty[0][indcc] == 0 or countPerDifficulty[nin][indcc].item()==0:
                        writer.add_scalar(test + '/AccuracyPer' + str(indcc) + '_' + str(nin) + 'spec', 0, epoch)
                    else:
                        writer.add_scalar(test + '/AccuracyPer' + str(indcc) + '_' + str(nin) + 'spec',
                                      (accuracyPerDifficulty[0][indcc].item() / countPerDifficulty[0][indcc].item()) +
                                      (accuracyPerDifficulty[nin][indcc].item() / countPerDifficulty[nin][indcc].item()), epoch)
                writer.add_scalar(test + '/countcorrectPer' +str(indcc)+'_'+ str(nin), accuracyPerDifficulty[nin][indcc].item(), epoch)
            writer.add_scalar(test + '/countPer' + str(indcc)+'_'+str(nin),  countPerDifficulty[nin][indcc], epoch)


def collect_accuracy_satistics(correctExist,countAllV,labels,indc,accuracyPerDifficulty,countPerDifficulty,correctLabels,correctLabelsTotal,correctExistLabTotal,outputs,correctList,correct,correctex,correctexList,PositiveList,NegativeList,Positivesumpleslst,Negativesumpleslist,correctTotal,correctExistTotal):
    Positive = (correctExist * countAllV).sum()
    Positivesumples = countAllV.sum()
    Negative = (correctExist * (1 - countAllV)).sum()
    Negativesumples = (1 - countAllV).sum()
    cconnt = 0
    for lab in labels:
        accuracyPerDifficulty[lab][indc] += correctExist[cconnt]
        countPerDifficulty[lab][indc] += 1
        cconnt += 1
    if indc == 0:
        correctLabelsTotal = correctLabels.float()
        correctExistLabTotal = correctExist.float()
    else:
        correctLabelsTotal += correctLabels.float()
        correctExistLabTotal += correctExist.float()

    if len(correctList) < len(outputs):
        correct.append(correctLabels.sum())
        correctex.append(correctExist.sum())

        correctList.append(correct[indc])
        correctexList.append(correctex[indc])
        PositiveList.append(Positive)
        NegativeList.append(Negative)
        Positivesumpleslst.append(Positivesumples)
        Negativesumpleslist.append(Negativesumples)
    else:
        PositiveList[indc] += Positive
        NegativeList[indc] += Negative
        Positivesumpleslst[indc] += Positivesumples
        Negativesumpleslist[indc] += Negativesumples
        correct[indc] += correctLabels.sum()
        correctex[indc] += correctExist.sum()
        correctList[indc] = correct[indc]
        correctexList[indc] = correctex[indc]
    if (indc+1) == len(outputs):
        correctTotal += (correctLabelsTotal > len(outputs) // 2).sum()
        correctExistTotal += (correctExistLabTotal > len(outputs) // 2).sum()
    return correctLabelsTotal, correctExistLabTotal, correctTotal, correctExistTotal



def eval(epoch,dataloder,test='TestRad', accbOld=0, printImage=False):
    net.eval()
    print(printImage)
    test_loss = 0.0  # cost function error
    correct = []
    correctex = []
    correctExistTotal=0.0
    correctTotal=0.0
    startind = 0
    endind = startind + 10
    test_loss = []
    test_lossEx=[]
    test_std = []
    correctList = []
    correctexList=[]
    lossAcc = []
    lossexAcc=[]
    lossSeg=0
    iouscore2=[]
    iouscore1=[]
    iouscore3 = []
    iouscore4 = []
    iouscore01=[]
    iouscore11=[]
    iouscore5 = []
    iouscore6 = []
    iouscore7 = []
    iouscore8 = []
    iouscore9=[]
    accuracyPerDifficulty=np.zeros((10, 6))
    countPerDifficulty =np.zeros((10, 6))
    PositiveList = []
    NegativeList = []
    Positivesumpleslst = []
    Negativesumpleslist = []

    # lossAcc=None
    countallconcut=None
    with torch.no_grad():
        bi=0
        for batch_index, (images, labels,bbbox, numbb) in enumerate(dataloder):
            # calc network output
            images = Variable(images)
            seglabel= (labels//28).long().cuda()#.unsqueeze(1)
            seglabel2 = (labels > 0).long().cuda()
            countAll =(labels > 0).sum(dim=1).sum(dim=1)
            seglabelMaxima, _ = torch.max(seglabel, dim=1)
            seglabelMaxima, _ = torch.max(seglabelMaxima, dim=1)

            countexx = 0
            labelsGroups = torch.zeros_like(seglabelMaxima)

            for labmax in seglabelMaxima:
                labelsGroups[countexx] = labels2Group[labmax]
                countexx += 1

            labelsGroup = Variable(labelsGroups * (countAll > 0).cuda().long())
            labels = Variable((seglabelMaxima ) * (countAll > 0).cuda().long())
            countAllV = Variable( (countAll > 0).cuda().long())
            labels = labels.long().cuda()
            images = images.float().cuda()
            imagesperm = images.permute(0, 3, 1, 2)

            outputs,maskarrer,_ = net(imagesperm)

            # collect performance
            out =outputs[-1]
            outputs=outputs[:-1]
            indc=0
            correctLabelsTotal=None
            correctExistLabTotal=None
            if labels.min()<0:
                labels[labels<0]=9

            for output in outputs:
                loss = loss_function(output[:, :4], labelsGroup).mean()
                lossex = loss_function(output[:, 10:12], countAllV).mean()
                if len(test_loss) < len(outputs):
                    lossAcc.append(loss.unsqueeze(0))
                    lossexAcc.append(lossex.unsqueeze(0))

                    test_loss.append(lossAcc[indc].mean())
                    test_lossEx.append(lossexAcc[indc].mean())

                else:
                    lossexAcc[indc] = torch.cat((lossexAcc[indc], lossex.unsqueeze(0)), 0)
                    lossAcc[indc]=torch.cat((lossAcc[indc], loss.unsqueeze(0)), 0)
                    test_loss[indc]=lossAcc[indc].mean()
                    test_lossEx[indc] = lossexAcc[indc].mean()

                _, preds = output[:, :4].max(1)
                _, predsT = output[:, endind:endind+2].max(1)
                correctLabels = preds.eq(labelsGroup)
                correctExist = predsT.eq(countAllV)

                # save attention mask
                if ((accbOld > 0.7 ) and epoch > 5) and indc==2 and printImage and (test=='TestRad'):  #and countAllV.sum() >= 2
                    counj = 0
                    for masked in maskarrer:
                        counj += 1
                        for i in range(0, masked.shape[0]):
                            lab=labels[i].item()
                            iscorrect=int(correctExist[i].float().item())
                            Maskedd = Image.fromarray(np.squeeze(masked[i, 0, :, :].cpu().numpy()))
                            Maskedd.save(
                                'ImagesRadGO/' + str(lab) + '/' + str(iscorrect) + '/' + str(epoch) + '_' + str(batch_index) + '_' + str(i) + '_' + str(
                                    counj) + 'Masked.tiff')
                            if counj <= 1:
                                labelsIm = Image.fromarray(
                                    np.squeeze(seglabel2[i, :, :] * 250).cpu().numpy().astype(np.uint8))
                                labelsIm.save(
                                    'ImagesRadGO/' + str(lab) + '/' + str(iscorrect) + '/' + str(epoch) + '_' + str(batch_index) + '_' + str(i) + 'label.tiff')
                                org = Image.fromarray(np.squeeze(imagesperm[i, 1, :, :].cpu().numpy()))
                                org.save(
                                    'ImagesRadGO/' + str(lab) + '/' + str(iscorrect) + '/' + str(epoch) + '_' + str(batch_index) + '_' + str(i) + 'org.tiff')

                correctLabelsTotal, correctExistLabTotal,correctTotal, correctExistTotal=collect_accuracy_satistics(correctExist, countAllV, labels, indc, accuracyPerDifficulty,
                                           countPerDifficulty, correctLabels, correctLabelsTotal, correctExistLabTotal,
                                           outputs, correctList, correct, correctex, correctexList, PositiveList,
                                           NegativeList, Positivesumpleslst, Negativesumpleslist, correctTotal,
                                           correctExistTotal)
                indc +=1



            lossSeg+=segloss(out[:,:10,:,:], seglabel)
            _,pred=out[:,:10,:,:].max(1)
            _, predTT = out[:, 10:12, :, :].max(1)

            temp=iou_pytorch(predTT == 0, seglabel2 == 0)
            temper=temp.item()
            iouscore01.append(temper)
            iouscore11.append(iou_pytorch(predTT == 1, seglabel2 == 1).item())
            if 1:
                iouscore4.append(iou_pytorch(pred == 4, seglabel == 4).item())
                iouscore3.append(iou_pytorch(pred == 3, seglabel == 3).item())
                iouscore2.append(iou_pytorch(pred == 2,seglabel == 2).item())
                iouscore1.append(iou_pytorch(pred == 1,seglabel == 1).item())
                iouscore5.append(iou_pytorch(pred == 5, seglabel == 5).item())
                iouscore6.append(iou_pytorch(pred == 6, seglabel == 6).item())
                iouscore7.append(iou_pytorch(pred == 7, seglabel == 7).item())
                iouscore8.append(iou_pytorch(pred == 8, seglabel == 8).item())
                iouscore9.append(iou_pytorch(pred == 9, seglabel == 9).item())

    print('Test set: Average lossSeg: {:.4f}, Accuracy seg: {:.4f}'.format(
        lossSeg / len(TesterRad_dataloder.dataset),
        avmean(iouscore2)
    ))
    print('Test set: Average lossSeg: {:.4f}, Accuracy seg: {:.4f}'.format(
        lossSeg / len(TesterRad_dataloder.dataset),
        avmean(iouscore3)
    ))
    print('Test set: Average lossSeg: {:.4f}, Accuracy seg: {:.4f}'.format(
        lossSeg / len(TesterRad_dataloder.dataset),
        avmean(iouscore4)
    ))
    print (accuracyPerDifficulty)
    print(countPerDifficulty)

    calcF1_and_TPR(PositiveList, Positivesumpleslst, Negativesumpleslist, NegativeList, test,epoch)

    generate_acc_satistic_per_difficulty(accuracyPerDifficulty, countPerDifficulty, test,epoch)


    writer.add_scalar(test + '/Average_lossSeg',lossSeg / len(dataloder.dataset), epoch)

    if 1:
        writer.add_scalar(test + '/test_seg1',  avmean(iouscore1) , epoch)
        writer.add_scalar(test + '/test_seg2',   avmean(iouscore2), epoch)
        writer.add_scalar(test + '/test_seg3', avmean(iouscore3), epoch)
        writer.add_scalar(test + '/test_seg4', avmean(iouscore4), epoch)
        writer.add_scalar(test + '/test_seg5', avmean(iouscore5), epoch)
        writer.add_scalar(test + '/test_seg7', avmean(iouscore6), epoch)
        writer.add_scalar(test + '/test_seg8', avmean(iouscore7), epoch)
        writer.add_scalar(test + '/test_seg8', avmean(iouscore8), epoch)
        writer.add_scalar(test + '/test_seg9', avmean(iouscore9), epoch)
    writer.add_scalar(test + '/test_seg00', avmean(iouscore01), epoch)
    writer.add_scalar(test + '/test_seg01', avmean(iouscore11), epoch)
    maxcor=0
    maxcorex=0
    for indc in range(0,len(correctList)):

        # add informations to tensorboard
        writer.add_scalar(test + '/Average lossEx' +str(indc), test_lossEx[indc] / len(dataloder.dataset), epoch)
        writer.add_scalar(test + '/AccuracyEx' +str(indc), correctexList[indc].float() / len(dataloder.dataset), epoch)
        writer.add_scalar(test + '/Average loss' + str(indc), test_loss[indc] / len(dataloder.dataset), epoch)
        writer.add_scalar(test + '/Accuracy' + str(indc), correctList[indc].float() / len(dataloder.dataset),
                          epoch)
        if maxcor<correctList[indc].float() / len(dataloder.dataset):
            maxcor=correctList[indc].float() / len(dataloder.dataset)


    writer.add_scalar(test + '/AccuracyTotal', float(correctTotal) / len(dataloder.dataset), epoch)
    writer.add_scalar(test + '/AccuracytExistTotal', float(correctExistTotal) / len(dataloder.dataset), epoch)
    if maxcor< float(correctTotal) / len(dataloder.dataset):
        maxcor= float(correctTotal) / len(dataloder.dataset)
    if maxcorex < float(correctExistTotal) / len(dataloder.dataset):
        maxcorex = float(correctExistTotal) / len(dataloder.dataset)
    writer.add_scalar(test + '/maxacc', maxcor, epoch)
    writer.add_scalar(test + '/maxaccEx', maxcorex, epoch)
    cormaxi = 0
    for cor in correctexList:
        if cor > cormaxi:
            cormaxi = cor
    return maxcorex, avmean(iouscore2), cormaxi.float() / len(valRad_dataloder.dataset)


def loadnet(chekpoint):
    optimizer_dict = None
    epoch_Start=0
    if chekpoint == '':
        net = efficientnet.efficientnet_b3(pretrained=True).cuda()  # resnext.resnext50_32x4d(pretrained=True).cuda()
    else:
        net = efficientnet.efficientnet_b3(pretrained=True).cuda()  # resnext.resnext50_32x4d().cuda()
        modelloaded = torch.load(chekpoint)
        model_dict = net.state_dict()
        updatedict = {}
        if 0:
            for k, v in modelloaded.items():
                if k in model_dict:
                    if len(v.shape) > 0:
                        if (model_dict[k].shape[0] != v.shape[0]):
                            print(model_dict[k].shape)
                            print(v.shape)
                        else:
                            updatedict[k] = v
                    else:
                        updatedict[k] = v
        if 0:
            pretrained_dict = {k: v for k, v in modelloaded['model_state_dict'].items() if
                               k in model_dict and (len(v.shape) == 0 or (model_dict[k].shape[0] == v.shape[0]))}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        else:
            net.load_state_dict(modelloaded['model_state_dict'])
        epoch_Start = modelloaded['epoch'] + 1
        optimizer_dict = modelloaded['optimizer_state_dict']
    return net,epoch_Start,optimizer_dict

def set_tensorboard_and_checkpoint():
    tn = settings.TIME_NOW
    tn = str.replace(tn, ':', ',')
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, tn)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    tn = settings.TIME_NOW
    tn = str.replace(tn, ':', ',')
    strpath = settings.LOG_DIR + "/" + args.net + "/" + tn
    writer = SummaryWriter(log_dir=strpath)
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    if hasattr(torch.autograd.Variable, 'grad_fn'):
        writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    return checkpoint_path , writer


def get_dataset():
    train_dir = 'HS_Data'
    train_list = '0.txt'
    Erd_Thc = 0

    cifar100_training_loader = DP.Dataset(
        train_dir + '/train2',  #

        'T2',  # 'All',#'T1c',#'Flair',#
        'LABEL',
        split='train2',
        train_dirr=train_dir,

        batch_size=args.b, shuffle=args.s, num_batches=20,
        num_workers=args.w, pin_memory=False, finalSize=256, multi=1, sever_per_arr=(18, 21, 25, 30, 35), TST_flag=True,
        ref_flg=True)  # 15, 18,
    train_dataloder = DataLoader(cifar100_training_loader, args.b, args.s)


    cifar100_training_loader = DP.Dataset(
        train_dir + '/train',  #

        'T2',  # 'All',#'T1c',#'Flair',#
        'LABEL',
        split='train',
        train_dirr=train_dir,
        # augmentation=DP.get_training_augmentation(),
        # preprocessing=DP.get_preprocessing(preprocessing_fn),
        batch_size=args.b, shuffle=args.s, num_batches=20,
        num_workers=args.w, pin_memory=False, finalSize=256, multi=1,
        sever_per_arr=(18, 21, 25, 30, 35))  # 15, 18,, 35, 41, 45
    train_dataloder3 = DataLoader(cifar100_training_loader, args.b, args.s)

    cifar100_training_loader2 = DP.Dataset(
        'MS_Data//train',  #

        'T2',  # 'All',#'T1c',#'Flair',#
        'LABEL',
        split='train',
        # augmentation=DP.get_training_augmentation(),
        # preprocessing=DP.get_preprocessing(preprocessing_fn),
        batch_size=args.b, shuffle=args.s, num_batches=20,
        num_workers=args.w, pin_memory=False, finalSize=256, ms__flg=True, gpu=True,
        train_dirr='MS_Data')  # sever_per_arr=(0, 21, 25, 30) # #(0, 21, 25, 30), #(0, 6, 59, 12), #(0, 6, 9, 12, 15, 18, 21, 25, 30)
    train_dataloder2 = DataLoader(cifar100_training_loader2, args.b, args.s)

    TesterRad = DP.Dataset(
        'HT_RData',

        'T2',  # 'T1c',#'ALL',#'T1c',#'T2',
        'LABEL',
        split='',
        # augmentation=DP.get_training_augmentation(),
        # preprocessing=DP.get_preprocessing(preprocessing_fn),
        batch_size=args.b, shuffle=False, num_batches=20,
        num_workers=args.w, pin_memory=False, finalSize=256, multi=1, gpu=True, train_dirr='HT_RData',
        TST_flag=True)

    TesterRad_dataloder = DataLoader(TesterRad, args.b, False)

    valRad = DP.Dataset(
        train_dir + '/val2',  #

        'T2',  # 'All',#'T1c',#'Flair',#
        'LABEL',
        split='val2',
        train_dirr=train_dir,
        # augmentation=DP.get_training_augmentation(),
        # preprocessing=DP.get_preprocessing(preprocessing_fn),
        batch_size=args.b, shuffle=args.s, num_batches=20,
        num_workers=args.w, pin_memory=False, finalSize=256, multi=1,
        sever_per_arr=(0, 6, 9, 12, 15, 18, 21, 25, 30), prob=0.5,
        TST_flag=True, ref_flg=True)  # 15, 18,

    valRad_dataloder = DataLoader(valRad, args.b, False)

    return train_dataloder, train_dataloder2, train_dataloder3,valRad_dataloder,TesterRad_dataloder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='efficentunetGenMultiORD2', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=3, help='warm up training phase')
    parser.add_argument('-useG', type=bool, default=True, help='Use generated lesions')
    parser.add_argument('-useGN', type=bool, default=True, help='Use generated lesions Live')
    parser.add_argument('-useOrg', type=bool, default=True, help='Use real lesions')
    parser.add_argument('-trainAll', type=bool, default=True, help='is to updates all parameters')
    parser.add_argument('-chekpoint', type=str, default='', help='checkpoint load')
    parser.add_argument('-lr', type=float, default=0.000841, help='initial learning rate') #0.000841 #0.0000281
    args = parser.parse_args()

    useOrg = args.useOrg
    useG =  args.useG
    useGN =  args.useGN
    chekpoint = args.chekpoint
    trainAll = args.trainAll

    net, epoch_Start, optimizer_dict=loadnet(chekpoint)

    train_dataloder, train_dataloder2, train_dataloder3, valRad_dataloder, TesterRad_dataloder = get_dataset()

    #set loss and optimiser
    loss_function = nn.CrossEntropyLoss(reduction='none')
    segloss=nn.CrossEntropyLoss()


    parameters = {param for name, param in net.named_parameters() if name.split('.')[0] != 'features'}

    fullPar = net.parameters()
    if trainAll == True:
        parameters = fullPar

    optimizer = torch.optim.RMSprop(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-5) #
    iter_per_epoch = len(train_dataloder.dataset)/(args.b*20)

    # create checkpoint folder to save model
    checkpoint_path, writer =set_tensorboard_and_checkpoint()



    best_accseg = 0.0
    best_accval = 0.0

    print("FinishPreProcess")
  #  acc, accseg = eval_trainingGenMul(0)
   # accTT, accsegTT = TestGen(0, 0)

    acc, accseg, cormaxx = eval(0, valRad_dataloder, test='eval', accbOld=best_accval)  #
    cormaxxx=cormaxx.item()

  #  best_accval = acc.item()
    #accTTR, accsegTTR = TestRadGen(0, best_accval)
    maxaccTotal=0
    maxaccTotalmax=0

    for epoch in range(epoch_Start, settings.EPOCH):

        if maxaccTotalmax < 0.85: #if best_accval>0.75 or maxaccTotalmax<0.85:
            maxaccTotal=train(epoch,best_accval,cormaxxx)
        else:
            maxaccTotal = train(epoch, maxaccTotalmax,cormaxxx)
        if maxaccTotalmax<maxaccTotal:
            maxaccTotalmax = maxaccTotal

        print(maxaccTotal)
        if  (epoch%2==1 or 1) :
            acc, accseg, cormaxi = eval(epoch,valRad_dataloder,test='eval',accbOld=best_accval) #

           # accTT, accsegTT =eval_trainingGen(epoch,best_accval)
            print(cormaxi)
            print(cormaxxx)
            accTTR, accsegTTR, cormaxiTest = eval(epoch,TesterRad_dataloder,test='TestRad',accbOld= best_accval,printImage=(cormaxxx*0.9<cormaxi) and (cormaxi>0.8))
            if cormaxxx<cormaxi:
                cormaxxx=cormaxi
       # break
            bsaved=False
            # start to save best performance model after learning rate decay to 0.01
            if best_accval < acc:
                bsaved=True

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_accval': acc
                    }, checkpoint_path.format(net=args.net, epoch=epoch, type='bacc'))
                best_accval = acc.item()
                print("Best acc")
                print(best_accval)
                continue



    writer.close()
