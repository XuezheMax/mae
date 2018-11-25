import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from torchvision.utils import save_image

from mae.data import load_datasets, get_batch, iterate_minibatches
from mae.modules import MAE

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--data', choices=['mnist', 'omniglot', 'cifar10', 'lsun'], help='data set', required=True)
parser.add_argument('--mode', choices=['generate', 'tsne', 'pca'], help='mode', required=True)
parser.add_argument('--seed', type=int, default=65537, metavar='S', help='random seed (default: 65537)')
parser.add_argument('--model_path', help='path for saving model file.', required=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

model_path = args.model_path
result_path = os.path.join(model_path, 'visualize')
if not os.path.exists(result_path):
    os.makedirs(result_path)

dataset = args.data
colorful = dataset in ['cifar10', 'lsun']
train_data, test_data, n_val = load_datasets(dataset)

train_index = np.arange(len(train_data))
test_index = np.arange(len(test_data))
np.random.shuffle(test_index)

mae = MAE.load(model_path, device)

mode = args.mode


def reconstruct(random_sample):
    n = 128
    data, _ = get_batch(test_data, test_index[:n])
    data = data.to(device)
    if not colorful:
        data = torch.ge(data, 0.5).float()

    recon_img, _ = mae.reconstruct(data, random_sample=random_sample)
    comparison = torch.cat([data, recon_img], dim=0).cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct.random.png' if random_sample else 'reconstruct.fixed.png'

    if colorful:
        save_image(comparison, os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))
    else:
        save_image(comparison, os.path.join(result_path, image_file), nrow=16)


def encode(visual_data, data_index):
    latent_codes = []
    labels = []
    num_insts = 0
    num_back = 0
    for i, (data, label) in enumerate(iterate_minibatches(visual_data, data_index, 200, False)):
        data = data.to(device)
        if not colorful:
            data = torch.ge(data, 0.5).float()
        batch_size = len(data)
        num_insts += batch_size
        z, _ = mae.sample_from_posterior(data, nsamples=50)
        # [batch, k, z_shape] -> [batch, z_shape] -> [batch, nz]
        z = z.mean(dim=1).view(z.size(0), -1)
        latent_codes.append(z)
        labels.append(label)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        log_info = '{}/{} ({:.0f}%)'.format(num_insts, len(data_index), 100. * num_insts / len(data_index))
        sys.stdout.write(log_info)
        sys.stdout.flush()
        num_back = len(log_info)
    return torch.cat(latent_codes, dim=0), torch.cat(labels, dim=0)


def generate_x():
    mae.eval()
    print("generating images:")
    reconstruct(random_sample=False)
    reconstruct(random_sample=True)
    sample_z, _ = mae.sample_from_proir(256, device=device)
    sample_x, _ = mae.decode(sample_z, random_sample=True)
    image_file = 'sample.png'
    if colorful:
        save_image(sample_x.cpu(), os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))
    else:
        save_image(sample_x.cpu(), os.path.join(result_path, image_file), nrow=16)


def tsne():
    assert dataset in ['mnist', 'cifar10']
    time_start = time.time()
    print('encoding:')
    latent_codes_train, labels_train = encode(train_data, train_index)
    latent_codes_test, labels_test = encode(test_data, test_index)

    latent_codes = torch.cat([latent_codes_train, latent_codes_test], dim=0).cpu().numpy()
    labels = torch.cat([labels_train, labels_test], dim=0).cpu().numpy()
    print('time: {:.1f}s'.format(time.time() - time_start))

    print('tSNE visualizing:')
    color_names = ['orangered', 'goldenrod', 'olivedrab', 'mediumseagreen', 'forestgreen', 'dodgerblue', 'steelblue', 'mediumslateblue', 'orchid', 'hotpink']

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(latent_codes)

    opacity = 0.2
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(10):
        color = colors[color_names[i]]
        ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], s=70, label=str(i), c=color, alpha=opacity, linewidths=0)
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    fig.savefig(os.path.join(result_path, 'tSNE.png'), bbox_inches='tight')


def pca():
    assert dataset in ['mnist', 'cifar10']
    time_start = time.time()
    print('encoding:')
    latent_codes_train, labels_train = encode(train_data, train_index)
    latent_codes_test, labels_test = encode(test_data, test_index)

    latent_codes = torch.cat([latent_codes_train, latent_codes_test], dim=0).cpu().numpy()
    labels = torch.cat([labels_train, labels_test], dim=0).cpu().numpy()
    print('time: {:.1f}s'.format(time.time() - time_start))

    print('PCA visualizing:')
    color_names = ['orangered', 'goldenrod', 'olivedrab', 'mediumseagreen', 'forestgreen', 'dodgerblue', 'steelblue', 'mediumslateblue', 'orchid', 'hotpink']

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(latent_codes)

    opacity = 0.2
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(10):
        color = colors[color_names[i]]
        ax.scatter(pca_results[labels == i, 0], pca_results[labels == i, 1], s=70, label=str(i), c=color, alpha=opacity, linewidths=0)
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    fig.savefig(os.path.join(result_path, 'pca.png'), bbox_inches='tight')


with torch.no_grad():
    if mode == 'generate':
        generate_x()
    elif mode == 'tsne':
        tsne()
    elif mode == 'pca':
        pca()
    else:
        raise ValueError('unknown mode: {}'.format(mode))
