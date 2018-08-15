import scipy.io
import numpy as np

import torch
from torchvision import datasets, transforms


def load_datasets(dataset, data_path=None):
    if dataset == 'omniglot':
        return load_omniglot()
    elif dataset == 'mnist':
        return load_mnist()
    elif dataset == 'lsun':
        return load_lsun(data_path)
    elif dataset == 'cifar10':
        return load_cifar10()
    else:
        raise ValueError('unknown data set %s' % dataset)


def load_omniglot():
    def reshape_data(data):
        return data.T.reshape((-1, 1, 28, 28))

    omni_raw = scipy.io.loadmat('data/omniglot/chardata.mat')

    train_data = reshape_data(omni_raw['data']).astype(np.float32)
    test_data = reshape_data(omni_raw['testdata']).astype(np.float32)
    return train_data, test_data, 2345


def load_mnist():
    train_data, _ = torch.load('data/mnist/processed/training.pt')
    test_data, _ = torch.load('data/mnist/processed/test.pt')

    train_data = train_data.float().div(255).unsqueeze(1)
    test_data = test_data.float().div(255).unsqueeze(1)

    return train_data.numpy(), test_data.numpy(), 2000


def load_lsun(data_path):
    imageSize = 32
    train_data = datasets.LSUN(data_path, classes=['bedroom_train'],
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    val_data = datasets.LSUN(data_path, classes=['bedroom_val'],
                             transform=transforms.Compose([
                                 transforms.Resize(imageSize),
                                 transforms.CenterCrop(imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

    return train_data, val_data


def load_cifar10():
    train_data = datasets.CIFAR10('data/cifar10', train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))
    test_data = datasets.CIFAR10('data/cifar10', train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
    return train_data, test_data
