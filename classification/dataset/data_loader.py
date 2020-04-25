import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
class data_loader:
    def __init__(self,args):
        self.args=args
        self.transform=transforms.Compose([
    # transforms.Scale(args.scale),
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
    def get_data_loader(self,trainset,testset):
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True,num_workers=self.args.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True,num_workers=self.args.num_workers)
        return train_loader,test_loader
    
    def load_mnist(self):
        args=self.args
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        minist_train=torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
        minist_test=torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)
        # 4,1,28,28
        return self.get_data_loader(minist_train,minist_test)
    
    def load_svhn(self):
        svhn_train = torchvision.datasets.SVHN('./data', split='train', transform=self.transform,
                                               target_transform=None,
                                               download=False)
        svhn_test = torchvision.datasets.SVHN('./data', split='test', transform=self.transform,
                                              target_transform=None,
                                              download=False)
        # 4,3,32,32
        return self.get_data_loader(svhn_train,svhn_test)
    
    def load_usps(self):
        args = self.args
        transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        usps_train = torchvision.datasets.USPS(root='./data', train=True, transform=transform,
                                                  download=True)
        usps_test = torchvision.datasets.USPS(root='./data', train=False, transform=transform,
                                                  download=True)
        # 4,1,16,16
        return self.get_data_loader(usps_train,usps_test)
