import torch
import torch.nn as nn
from dataset.data_loader import *
import torch.optim as optim
import torch.nn.functional as F
import argparse
from model import model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
def get_entropy(r1):
    r1 = F.softmax(r1)
    return -torch.mean(torch.log(torch.mean(r1,0)+1e-6))
def get_dis(r1,r2,mode):
    if mode=='ad_drop':
        r1=F.softmax(r1)
        return  - torch.mean(r1 * torch.log(r1 + 1e-6))
    else:
        return torch.mean(torch.abs(F.softmax(r1)-F.softmax(r2)))

# stepA train both classifiers and generator to classify the source samples correctly.
def save_model(best,acc):
    if best < acc:
        print('saving')
        best = acc
        torch.save({
            "G_state_dict": net_G.state_dict(),
            "C1_state_dict": net_D1.state_dict(),
            "C2_state_dict": net_D2.state_dict()
        }, args.model_name
        )
    return best

def set_zero_grad(netG,netD1,netD2):
    netG.zero_grad()
    netD1.zero_grad()
    netD2.zero_grad()
def  train(args,source_trainset,target_trainset):
    criterion=nn.CrossEntropyLoss()
    opt_g, opt_c1, opt_c2=set_optimizer(net_G,net_D1,net_D2,args.opt,lr=args.lr,momentum=args.momentum)
    # net_D1.set_lambda(1.0)
    # net_D2.set_lambda(1.0)
    best=0
    torch.cuda.manual_seed(1)
    for epoch in range(args.epochs):
        for i,data in enumerate(source_trainset):
            if i * args.batch_size > 30000:
                break
            if data[0].size(0)!=args.batch_size:
                break
            feature = net_G(data[0].to(args.device),args.mode)
            c1 = net_D1(feature,args.mode)
            c2 = net_D2(feature,args.mode)

            target=next(iter(target_trainset))[0].to(args.device)
            f_t = net_G(target, args.mode)
            c1_t = net_D1(f_t, args.mode)
            c2_t = net_D2(f_t, args.mode)
            label = data[1].to(args.device)
            entropy=get_entropy(c1_t)+get_entropy(c2_t)
            lossA = criterion(c1, label) + criterion(c2, label)+0.01*entropy

            # print('stepA' )
            lossA.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()
            set_zero_grad(net_G,net_D1,net_D2)

            # print('stepB' )
            if target.size(0)!=args.batch_size:
                break
            feature = net_G(data[0].to(args.device),args.mode).detach()
            c1 = net_D1(feature,args.mode)
            c2 = net_D2(feature,args.mode)
            label = data[1].to(args.device)
            f_t=net_G(target,args.mode).detach()
            c1_t=net_D1(f_t,args.mode)
            if args.mode == 'normal':
                c2_t=net_D2(f_t,args.mode)
            # maximize discrepancy
            # fix_net(net_G,False)
                l_adv=get_dis(c1_t,c2_t,args.mode)

                entropy = get_entropy(c1_t) + get_entropy(c2_t)

                loss_B = criterion(c1, label) + criterion(c2, label)-l_adv+0.01*entropy
            else:
                c2_t = net_D1(f_t, args.mode)
                loss_B = criterion(c1, label)
                l_adv = get_dis(c1_t, c2_t, args.mode)
                loss_B -= l_adv
            loss_B.backward()
            opt_c1.step()
            opt_c2.step()
            set_zero_grad(net_G,net_D1,net_D2)
            # step C
            loss_C=0
            for itr_g in range(args.gen_epoch):
                f_t = net_G(target,args.mode)
                c1_t = net_D1(f_t,args.mode,reverse=False)
                if args.mode == 'normal':
                    c2_t = net_D2(f_t,args.mode,reverse=False)
                    # loss_C = -get_dis(c1_t, c2_t,args.mode)
                    # !!!change reverse to False and loss to positive comparing to the digit model
                    loss_C = get_dis(c1_t, c2_t,args.mode)
                else:
                    c2_t = net_D1(f_t, args.mode, reverse=False)
                    loss_C = get_dis(c1_t, c2_t, args.mode)
                loss_C.backward()
                opt_g.step()
            set_zero_grad(net_G,net_D1,net_D2)

            # fix_net(net_D1, True)
            # fix_net(net_D2, True)
            if i%200==0 :
                print(lossA,loss_B,loss_C)
                acc=test(args, target_trainset)
                best=save_model(best, acc)
                print('best:',best)

        if epoch==args.save_epoch-1:
            print('epochs:',epoch)
            acc=test(args, target_trainset)
            best=save_model(best,acc)



def set_optimizer(G,C1,C2, algorithm='SGD', lr=0.001, momentum=0.9):
    if algorithm == 'SGD':
        opt_g = optim.SGD(G.parameters(),
                               lr=lr, weight_decay=0.0005,
                               momentum=momentum)

        opt_c1 = optim.SGD(C1.parameters(),
                                lr=lr, weight_decay=0.0005,
                                momentum=momentum)
        opt_c2 = optim.SGD(C2.parameters(),
                                lr=lr, weight_decay=0.0005,
                                momentum=momentum)
        return opt_g, opt_c1, opt_c2
    if algorithm == 'Adam':
        opt_g = optim.Adam(G.parameters(),
                                lr=lr, weight_decay=0.0005)

        opt_c1 = optim.Adam(C1.parameters(),
                                 lr=lr, weight_decay=0.0005)
        opt_c2 = optim.Adam(C2.parameters(),
                                 lr=lr, weight_decay=0.0005)
        return opt_g,opt_c1,opt_c2

def test(args,target_test):
    print('testing....')
    # ckp = torch.load(args.model_name)
    # net_G.load_state_dict(ckp['G_state_dict'])
    # net_D1.load_state_dict(ckp['C1_state_dict'])
    # net_D2.load_state_dict(ckp['C2_state_dict'])
    correct0=0.0
    correct1=0.0
    correct2=0.0
    sum=0.0
    # class_right=[[0 for i in range(10)]]
    # class_count=[[0 for i in range(10)]]
    correct = list(0. for i in range(13))
    total = list(0. for i in range(13))
    for i,target_data in enumerate(target_test):
            if i * args.batch_size > 5000:
                break
            target=target_data[0].to(args.device)
            if target.size(0)!=args.batch_size:
                break
            label=target_data[1].to(args.device)
            f=net_G(target,args.mode)
            c1=(net_D1(f,args.mode))
            r1=c1.data.max(1)[1]
            c2=(net_D2(f,args.mode))
            r2=c2.data.max(1)[1]
            r3=(c1+c2).data.max(1)[1]
            # print(r1,label.data)
            # print(label[label==3])
            correct0+=(r1.eq(label.data).sum()).cpu()
            correct1+=(r2.eq(label.data).sum()).cpu()
            correct2+=(r3.eq(label.data).sum()).cpu()
            sum+=target.size(0)
            res=r1==label
            for label_idx in range(13):
                label_single = label[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1
    print(float(correct0/sum),float(correct1/sum),float(correct2/sum))
    for acc_idx in range(13):
        try:
            acc = correct[acc_idx] / total[acc_idx]
            print(acc)
        except:
            acc = 0

    return max(float(correct0/sum),float(correct1/sum),float(correct2/sum))
import os
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int, metavar='N')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='N')
    parser.add_argument('--opt', default='SGD', type=str, metavar='N')
    parser.add_argument('--mode', default='ad_drop', type=str, metavar='N')
    # parser.add_argument('--mode', default='normal', type=str, metavar='N')
    parser.add_argument('--model_name', default='result.pth.tar', type=str, metavar='N')
    parser.add_argument('--img_size', default=32, type=int, metavar='N')
    parser.add_argument('--gen_epoch', default=4, type=int, metavar='N')
    parser.add_argument('--save_epoch', default=50, type=int, metavar='N')
    # self.source = 'mnist'
    # self.target = 'svnh'
    args = parser.parse_args()
    args.model_name=args.mode+'/'+args.model_name
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # args=argument()
    net_G=model.Feature().to(args.device)
    net_D1=model.Predictor().to(args.device).apply(weights_init)
    net_D2=model.Predictor().to(args.device).apply(weights_init)

    if os.path.exists(args.model_name):
        print('loading....')
        ckp = torch.load(args.model_name,map_location=args.device)
        net_G.load_state_dict(ckp['G_state_dict'])
        net_D1.load_state_dict(ckp['C1_state_dict'])
        net_D2.load_state_dict(ckp['C2_state_dict'])
    source_trainset = get_loader(args,'train')
    targetset=get_loader(args,'validation')
    train(args,source_trainset,targetset)
    # test(args,targetset)




