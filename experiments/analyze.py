import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import json
import argparse
import random
import numpy as np

import torch
from torch import optim
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_

from mae.data import load_datasets
from mae.modules import MAE

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--data', choices=['mnist', 'omniglot', 'cifar10', 'lsun'], help='data set', required=True)
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--model_path', help='path for saving model file.', required=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')


def get_batch(data, indices):
    imgs = []
    labels = []
    for index in indices:
        img, label = data[index]
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs, dim=0), torch.LongTensor(labels)


model_path = args.model_path
result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

dataset = args.data
train_data, test_data, n_val = load_datasets(dataset)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
val_index = train_index[-n_val:]
train_index = train_index[:-n_val]

test_index = np.arange(len(test_data))
np.random.shuffle(test_index)

mae = MAE.load(model_path, device)

def reconstruct(random_sample):
    mae.eval()
    n = 128
    data, _ = get_batch(test_data, test_index[:n])
    data = data.to(device)

    recon_img, _ = mae.reconstruct(data, random_sample=random_sample)
    comparison = torch.cat([data, recon_img], dim=0).cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct.png'
    save_image(comparison, os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))


with torch.no_grad():
    reconstruct(random_sample=False)
    sample_z, (z_normal, logdet) = mae.sample_from_proir(256, device=device)
    sample_x, _ = mae.decode(sample_z, random_sample=True)
    image_file = 'sample.png'
    save_image(sample_x.cpu(), os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))
