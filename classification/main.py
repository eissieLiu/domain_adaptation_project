import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from model import model


def train(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    ############################################################

    net_G=model.Feature()
    net_D1=model.Predictor()
    net_D2=model.Predictor()
    criterion=nn.CrossEntropyLoss()
    opt_g, opt_c1, opt_c2=set_optimizer(net_G,net_D1,net_D2,args.opt)
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
            feature = net_G(data[0].to(args.device))
            c1 = net_D1(feature)
            c2 = net_D2(feature)
            label = data[1].to(args.device)
            f_t=net_G(target)
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
            test(args, target_test)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    os.environ['MASTER_PORT'] = '8888'                      #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################