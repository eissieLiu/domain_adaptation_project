import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP
from model import model
from dataset.data_loader_multi_gpu import *
from model import model

from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
class test_multi():
    def __init__(self,args):
        self.args=args
    def get_dis(self,r1,r2):
        return torch.mean(torch.abs(r1-r2))

    def get_data(self,data_loader,data):
        if data== 'svnh':
            return data_loader.load_svhn()
        if data== 'mnist':
            return data_loader.load_mnist()

    # stepA train both classifiers and generator to classify the source samples correctly.

    def set_zero_grad(self,netG,netD1,netD2):
        netG.zero_grad()
        netD1.zero_grad()
        netD2.zero_grad()
    def fix_net(self,net,value):
        for itr_n in net.parameters():
            itr_n.requires_grad=value

    def setup(self,rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'

        # initialize the process group
        # dist.init_process_group("gloo", rank=rank, world_size=world_size)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(42)
    def fix_net(self,net,value):
        for itr_net in net.parameters():
            itr_net.requires_grad=value
    def  train(self,gpu,args):
        # print(args)
        # print(rank,world_size)
        # args=self.args
        # args.rank=rank
        # args.world_size=world_size
        # self.setup(rank,world_size)
        # n = torch.cuda.device_count()
        rank = args.nr * args.gpus + gpu
        args.rank=rank
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
        device_ids=list([gpu])
    # device_ids = list(range(rank * n, (rank + 1) * n))
        args.device=gpu
        # print(gpu)
        # print(device_ids)
        # args.rank=rank

        data_loader=data_loader_multi_gpu(self.args)
        ############################################################
        source_trainset, source_testset = self.get_data(data_loader,args.source)
        target_trainset, target_test = self.get_data(data_loader,args.target)
        net_G = model.Feature().to(args.device)
        net_D1 = model.Predictor().to(args.device)
        net_D2 = model.Predictor().to(args.device)
        # torch.cuda.set_device(gpu)
        net_G = torch.nn.parallel.DistributedDataParallel(net_G,
                                                          device_ids=device_ids)
        net_D1 = torch.nn.parallel.DistributedDataParallel(net_D1,
                                                          device_ids=device_ids)
        net_D2 = torch.nn.parallel.DistributedDataParallel(net_D2,
                                                          device_ids=device_ids)
        criterion=nn.CrossEntropyLoss()
        opt_g, opt_c1, opt_c2=self.set_optimizer(net_G,net_D1,net_D2,args.opt)
        for epoch in range(args.epochs):
            for i,data in enumerate(source_trainset):
                img=data[0].to(args.device)
                if args.source=='mnist':
                    img=img.expand(args.batch_size,3,args.img_size,args.img_size)
                feature = net_G(img.to(args.device))
                c1 = net_D1(feature)
                c2 = net_D2(feature)
                label = data[1].to(args.device)
                lossA = criterion(c1, label) + criterion(c2, label)

                # print('stepA' )
                lossA.backward()
                opt_g.step()
                opt_c1.step()
                opt_c2.step()
                self.set_zero_grad(net_G,net_D1,net_D2)

                print('stepB' )
                self.fix_net(net_G,False)
                target=next(iter(target_trainset))[0].to(args.device)
                # target=target.cuda(non_blocking=True)
                if args.target=='mnist':
                    target=target.expand(args.batch_size,3,args.img_size,args.img_size)
                feature = net_G(target).detach()
                c1 = net_D1(feature)
                c2 = net_D2(feature)
                label = data[1].to(args.device)
                f_t=net_G(target)
                c1_t=net_D1(f_t)
                c2_t=net_D2(f_t)
                # maximize discrepancy
                # fix_net(net_G,False)
                l_adv=self.get_dis(c1_t,c2_t)
                loss_B = criterion(c1, label) + criterion(c2, label)-l_adv
                loss_B.backward(retain_graph=True)
                opt_c1.step()
                opt_c2.step()
                print('q')
                self.set_zero_grad(net_G,net_D1,net_D2)
                opt_g.step()
                print('stepC')

                self.fix_net(net_G,True)
                self.fix_net(net_D1, False)
                self.fix_net(net_D2, False)

                loss_C=0
                for itr_g in range(args.gen_epoch):
                    f_t = net_G(target)
                    c1_t = net_D1(f_t)
                    c2_t = net_D2(f_t)
                    loss_C = self.get_dis(c1_t, c2_t)
                    loss_C.backward()
                    opt_g.step()

                self.fix_net(net_D1, True)
                self.fix_net(net_D2, True)
                print(lossA,loss_B,loss_C)
                if i%500==0:
                    print(lossA,loss_B,loss_C)
            if epoch%args.save_epoch==0:
                print('epochs:',epoch)
                print('saving')
                torch.save({
                    "G_state_dict": net_G.state_dict(),
                    "C1_state_dict": net_D1.state_dict(),
                    "C2_state_dict": net_D2.state_dict()
                }, "result.pth.tar"
                )
                # self.test(args, target_test)


    def set_optimizer(self,G,C1,C2, algorithm='SGD', lr=0.001, momentum=0.9):
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

    # def test(self,args,target_test):
    #     print('testing....')
    #     # ckp = torch.load("stepA_result.pth.tar")
    #     # net_G.load_state_dict(ckp['G_state_dict'])
    #     # net_D1.load_state_dict(ckp['C1_state_dict'])
    #     # net_D2.load_state_dict(ckp['C2_state_dict'])
    #     correct0=0.0
    #     correct1=0.0
    #     correct2=0.0
    #     sum=0.0
    #     for i,target_data in enumerate(target_test):
    #             target=target_data[0].to(args.device)
    #             if args.target=='mnist':
    #                 target=target.expand(target.size(0),3,args.img_size,args.img_size)
    #             label=target_data[1].to(args.device)
    #             f=net_G(target)
    #             c1=(net_D1(f))
    #             r1=c1.data.max(1)[1]
    #             c2=(net_D2(f))
    #             r2=c2.data.max(1)[1]
    #             r3=(c1+c2).data.max(1)[1]
    #             correct0+=(r1.eq(label.data).sum()).cpu()
    #             correct1+=(r2.eq(label.data).sum()).cpu()
    #             correct2+(r3.eq(label.data).sum()).cpu()
    #             sum+=target.size(0)
    #     print(float(correct0/sum),float(correct1/sum),float(correct2/sum))
    def main(self,args):


        args.world_size = args.gpus * args.nodes

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        # print(args.world_size)
        mp.spawn(self.train, nprocs=args.gpus, args=(args,),join=True)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N')
    parser.add_argument('--opt', default='Adam', type=str, metavar='N')
    parser.add_argument('--img_size', default=32, type=int, metavar='N')
    parser.add_argument('--gen_epoch', default=4, type=int, metavar='N')
    parser.add_argument('--save_epoch', default=50, type=int, metavar='N')
    parser.add_argument('--source', default='svnh', type=str, metavar='N')
    parser.add_argument('--target', default='mnist', type=str, metavar='N')
    parser.add_argument('--rank', default=0, type=int, metavar='N')

    args = parser.parse_args()
    tm=test_multi(args)
    tm.main(args)
    # args=argument()
    # net_G=model.Feature().to(args.device)
    # net_D1=model.Predictor().to(args.device)
    # net_D2=model.Predictor().to(args.device)
    # if os.path.exists('stepA_result.pth.tar'):
    #     print('loading....')
    #     ckp = torch.load("stepA_result.pth.tar",map_location=args.device)
    #     net_G.load_state_dict(ckp['G_state_dict'])
    #     net_D1.load_state_dict(ckp['C1_state_dict'])
    #     net_D2.load_state_dict(ckp['C2_state_dict'])
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # 5) 封装
    #     net_G = torch.nn.parallel.DistributedDataParallel(model.Feature(),
    #                                                       device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank)
    #     net_D1 = torch.nn.parallel.DistributedDataParallel(model.Predictor(),
    #                                                       device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank)
    #     net_D2 = torch.nn.parallel.DistributedDataParallel(model.Predictor(),
    #                                                       device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank)
    # source_trainset, source_testset = get_data(args.source)
    # target_trainset,target_test=get_data(args.target)
    # train(args,source_trainset,target_trainset,target_test)
    # test(args,target_test)




