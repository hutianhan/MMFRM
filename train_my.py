import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random

from data import *
from MMFRM import *
from mobilefacenet import *


def save_valaccuracylist(content):
    filename="result_valaccuracylist.txt"
    with open(filename,mode='w') as fp:
        fp.write(str(content))

def save_y_true(content):
    filename="y_true.txt"
    with open(filename,mode='w') as fp:
        fp.write(str(content))

def save_y_pred(content):
    filename="y_pred.txt"
    with open(filename,mode='w') as fp:
        fp.write(str(content))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default=r'D:\PW\FER\datasets\raf-basic', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.05, help='Drop out rate.')
    return parser.parse_args()
    
def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
        

if __name__ == "__main__":
    weight_path=r'models\mypath.pth'

    if not torch.cuda.is_available():
        from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    imagenet_pretrained = True
    # network =RestNet18()
    #network=EfficientNetB0(in_channels=3, n_classes=7)
    #network = MobileNetV1()
    # network=DenseNet121()
    # network = My_VGG(num_classes=7)

    network = MMFRM().to(device)

    networkMobile=MobileFaceNet([112, 112]).to(device)
    weight_path_mobile = r'models\mobileface.pth'
    if os.path.exists(weight_path_mobile):
        networkMobile.load_state_dict(torch.load(weight_path_mobile))
        print('successful load weight！')
    else:
        print('not successful load weight')

    optimizer = torch.optim.Adam(network.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss()


    if not imagenet_pretrained:
        for m in network.modules():
            initialize_weight_goog(m)

    train_dataset = RafDataSet(args.raf_path, phase='train', basic_aug=True)
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    val_dataset = RafDataSet(args.raf_path, phase='test')
    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    beta = args.beta
    valaccuracylist = []

    y_true = []
    y_pred = []

    for t in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        network.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz * beta)
            optimizer.zero_grad()

            imgs = imgs.to(device)
            feature1, feature2, feature3,_,_ = networkMobile(imgs)
            attention_weights, outputs = network(imgs,feature1, feature2, feature3)


            sm = torch.softmax(outputs, dim=1)
            Pmax, predicted_labels = torch.max(sm, 1)
            a0=a1=a2=a3=a4=a5=a6 = []
            c0=c1=c2=c3=c4=c5=c6 = 0.0001
            s0 =s1=s2=s3=s4=s5=s6= 0.0
            sq0=sq1=sq2=sq3=sq4=sq5=sq6 = 0.0
            for i  in range(batch_sz):

                if targets[i]==0  :
                    s0=s0+sm[i][0]
                    c0=c0+1
                    a0.append(sm[i][0])

                if targets[i]==1  :
                    s1=s1+sm[i][1]
                    c1=c1+1
                    a0.append(sm[i][1])

                if targets[i] == 2:
                    s2 = s2 + sm[i][2]
                    c2 = c2 + 1
                    a0.append(sm[i][2])

                if targets[i] == 3:
                    s3= s3 + sm[i][3]
                    c3 = c3 + 1
                    a0.append(sm[i][3])
                if targets[i] == 4:
                    s4= s4 + sm[i][4]
                    c4 = c4 + 1
                    a0.append(sm[i][4])
                if targets[i] == 5:
                    s5= s5 + sm[i][5]
                    c5 = c5 + 1
                    a0.append(sm[i][5])
                if targets[i] == 6:
                    s6= s6 + sm[i][6]
                    c6 = c6 + 1
                    a0.append(sm[i][6])

            aver0 = s0 / c0
            aver1 = s1 / c1
            aver2= s2 / c2
            aver3 = s3 / c3
            aver4= s4/ c4
            aver5 = s5 / c5
            aver6= s6/ c6
            for j  in a0:
                sq0=sq0+math.sqrt(pow(j-aver0,2))
            for j  in a1:
                sq1=sq1+math.sqrt(pow(j-aver1,2))
            for j  in a2:
                sq2=sq2+math.sqrt(pow(j-aver2,2))
            for j in a3:
                sq3 = sq3 + math.sqrt(pow(j - aver3, 2))
            for j  in a4:
                sq4=sq4+math.sqrt(pow(j-aver4,2))
            for j  in a5:
                sq5=sq5+math.sqrt(pow(j-aver5,2))
            for j  in a6:
                sq6=sq6+math.sqrt(pow(j-aver6,2))
            RR_loss2=(sq0+sq1+sq2+sq3+sq4+sq5+sq6)/64


            targets = targets.to(device)
            if  t>50:
                loss = 0.9*criterion(outputs, targets) + 0.2*RR_loss2
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss

            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            sm = torch.softmax(outputs, dim=1)

        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (t, acc, running_loss))

        network.eval()
        y_vals = []
        predicteds = []

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            network.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):

                imgs=imgs.to(device)
                feature1, feature2, feature3,_,_ = networkMobile(imgs)
                _, outputs = network(imgs,feature1, feature2, feature3)

                targets = targets.to(device)

                y_vals += list(targets.cpu().numpy())

                targets = targets.to(device)
                if t > 50:
                    loss = 0.9 * criterion(outputs, targets) + 0.2 * RR_loss2
                else:
                    loss = criterion(outputs, targets)

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                predicteds += list(predicts.cpu().numpy())
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            running_loss = running_loss / iter_cnt
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (t, acc, running_loss))


            if t > 5 and acc > max(valaccuracylist):
                max_accuracy = acc
                max_epoch = t
                print('--------------------max_accuracy=', max_accuracy)
                print('--------------------max_accuracy=', max_epoch)
                torch.save(network.state_dict(), weight_path)
                print('save successfully!')

            valaccuracylist.append(acc)





