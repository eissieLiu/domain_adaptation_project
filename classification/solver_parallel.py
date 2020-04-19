import torch
import torch.nn as nn
from dataset.data_loader import *
import torch.optim as optim
import torch.nn.functional as F
from model import model
class argument:
    def __init__(self):
        self.batch_size=128
        self.num_workers=4
        self.opt='Adam'
        self.epochs=2000
        self.img_size=32
        self.gen_epoch=4
        self.save_epoch=50
        self.device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.source='svnh'
        self.target='mnist'
    def set_source(self,s):
        self.source=s
def get_dis(r1,r2):
    return torch.mean(torch.abs(F.softmax(r1)-F.softmax(r2)))

def get_data(data):
    if data== 'svnh':
        return data_loader.load_svhn()
    if data== 'mnist':
        return data_loader.load_mnist()

# stepA train both classifiers and generator to classify the source samples correctly.

def set_zero_grad(netG,netD1,netD2):
    netG.zero_grad()
    netD1.zero_grad()
    netD2.zero_grad()
def  train(args,source_trainset,target_trainset,target_test):
    criterion=nn.CrossEntropyLoss()
    net_G = model.Feature().to(args.device)
    net_D1 = model.Predictor().to(args.device)
    net_D2 = model.Predictor().to(args.device)
    if os.path.exists('stepA_result.pth.tar'):
        print('loading....')
        ckp = torch.load("stepA_result.pth.tar", map_location=args.device)
        net_G.load_state_dict(ckp['G_state_dict'])
        net_D1.load_state_dict(ckp['C1_state_dict'])
        net_D2.load_state_dict(ckp['C2_state_dict'])
    net_G=torch.nn.DataParallel(net_G, device_ids=[0,1])
    net_D1=torch.nn.DataParallel(net_D1, device_ids=[0,1])
    net_D2=torch.nn.DataParallel(net_D2, device_ids=[0,1])
    opt_g, opt_c1, opt_c2=set_optimizer(net_G,net_D1,net_D2,args.opt)

    torch.cuda.manual_seed(1)
    for epoch in range(args.epochs):
        for i,data in enumerate(source_trainset):

            if args.source=='mnist':
                data[0]=data[0].expand(args.batch_size,3,args.img_size,args.img_size)
            feature = net_G(data[0].to(args.device))
            c1 = net_D1(feature)
            c2 = net_D2(feature)
            label = data[1].to(args.device)
            lossA = criterion(c1, label) + criterion(c2, label)

            # print('stepA' )
            lossA.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()
            set_zero_grad(net_G,net_D1,net_D2)

            # print('stepB' )
            target=next(iter(target_trainset))[0].to(args.device)
            if args.target=='mnist':
                target=target.expand(args.batch_size,3,args.img_size,args.img_size)
            feature = net_G(data[0].to(args.device)).detach()
            c1 = net_D1(feature)
            c2 = net_D2(feature)
            label = data[1].to(args.device)
            f_t=net_G(target).detach()
            c1_t=net_D1(f_t)
            c2_t=net_D2(f_t)
            # maximize discrepancy
            # fix_net(net_G,False)
            l_adv=get_dis(c1_t,c2_t)
            loss_B = criterion(c1, label) + criterion(c2, label)-l_adv
            loss_B.backward()
            opt_c1.step()
            opt_c2.step()
            set_zero_grad(net_G,net_D1,net_D2)
            # print('stepC')

            # fix_net(net_G,True)
            # fix_net(net_D1, False)
            # fix_net(net_D2, False)

            loss_C=0
            for itr_g in range(args.gen_epoch):
                f_t = net_G(target)
                c1_t = net_D1(f_t)
                c2_t = net_D2(f_t)
                loss_C = get_dis(c1_t, c2_t)
                loss_C.backward()
                opt_g.step()

            # fix_net(net_D1, True)
            # fix_net(net_D2, True)
            if i%100==0:
                print(lossA,loss_B,loss_C)
                print('saving')
                torch.save({
                    "G_state_dict": net_G.state_dict(),
                    "C1_state_dict": net_D1.state_dict(),
                    "C2_state_dict": net_D2.state_dict()
                }, "A_result.pth.tar"
                )
                test(args,net_G,net_D1,net_D2, target_test)
        # if epoch%args.save_epoch==0:
        #     print('epochs:',epoch)
        #     print('saving')
        #     torch.save({
        #         "G_state_dict": net_G.state_dict(),
        #         "C1_state_dict": net_D1.state_dict(),
        #         "C2_state_dict": net_D2.state_dict()
        #     }, "stepA_result.pth.tar"
        #     )
        #     test(args, target_test)


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

def test(args,net_G,net_D1,net_D2,target_test):
    print('testing....')
    # ckp = torch.load("A_result.pth.tar")
    # net_G.load_state_dict(ckp['G_state_dict'])
    # net_D1.load_state_dict(ckp['C1_state_dict'])
    # net_D2.load_state_dict(ckp['C2_state_dict'])
    correct0=0.0
    correct1=0.0
    correct2=0.0
    sum=0.0
    for i,target_data in enumerate(target_test):
            target=target_data[0].to(args.device)
            if args.target=='mnist':
                target=target.expand(target.size(0),3,args.img_size,args.img_size)
            label=target_data[1].to(args.device)
            f=net_G(target)
            c1=(net_D1(f))
            r1=c1.data.max(1)[1]
            c2=(net_D2(f))
            r2=c2.data.max(1)[1]
            r3=(c1+c2).data.max(1)[1]
            correct0+=(r1.eq(label.data).sum()).cpu()
            correct1+=(r2.eq(label.data).sum()).cpu()
            correct2+(r3.eq(label.data).sum()).cpu()
            sum+=target.size(0)
    print(float(correct0/sum),float(correct1/sum),float(correct2/sum))
import os
if __name__=="__main__":
    args=argument()

    data_loader=data_loader(args=args)
    source_trainset, source_testset = get_data(args.source)
    target_trainset,target_test=get_data(args.target)
    train(args,source_trainset,target_trainset,target_test)
    # test(args,target_test)




