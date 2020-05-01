from torchvision import datasets,transforms
import torch
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def get_loader(args,data='train'):
    dataset=datasets.ImageFolder(data+'/',data_transforms)
    print(dataset.class_to_idx)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dl