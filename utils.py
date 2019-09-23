import os
import sys
import torchvision
import torchvision.transforms as transform

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler


def get_network(args, use_gpu=True):
    if args.net == 'vgg16':
        from model.vgg import vgg16_bn
        net = vgg16_bn()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()
    return net


def get_training_dataload(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True, download=True):
    transform_train = transform.Compose([
        transform.RandomCrop(32, padding=4),
        transform.RandomHorizontalFlip(),
        transform.RandomRotation(15),
        transform.ToTensor(),
        transform.Normalize(mean, std)
    ])

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    cifar100_training = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(cifar100_training, shuffle=shuffle, num_workers=num_workers,
                                          batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(data_dir, mean, std, batch_size=16, num_workers=2, shuffle=True, download=True):
    transform_test = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

# 自定义了学习率调度器
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
