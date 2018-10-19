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

from mae.data import load_datasets, iterate_minibatches, get_batch
from mae.modules import MAE
from mae.modules import exponentialMovingAverage

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--data', choices=['cifar10', 'lsun_bedroom', 'lsun_tower', 'lsun_church_outdoor'], help='data set', required=True)
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--opt', choices=['adam', 'adamax'], help='optimization method', default='adam')
parser.add_argument('--eta', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--gamma', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--free-bits', type=float, default=0.0, metavar='N', help='free bits used in training.')
parser.add_argument('--polyak', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('--schedule', type=int, default=50, help='schedule for learning rate decay')
parser.add_argument('--model_path', help='path for saving model file.', required=True)
parser.add_argument('--data_path', help='path for data file.', default=None)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

dataset = args.data
imageSize = 32 if dataset == 'cifar10' else 64
nc = 3
nx = imageSize * imageSize * nc
training_k = 1
test_k = 5
k = 512
eta = args.eta
gamma = args.gamma
free_bits = args.free_bits

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(result_path):
    os.makedirs(result_path)

train_data, test_data, n_val = load_datasets(dataset, args.data_path)

train_index = np.arange(len(train_data))
np.random.shuffle(train_index)
val_index = train_index[-n_val:]
train_index = train_index[:-n_val]

test_index = np.arange(len(test_data))
np.random.shuffle(test_index)

if dataset.startswith('lsun'):
    # for lsun data, repeat val and test data five times for random corps.
    val_index = np.concatenate([val_index, val_index, val_index, val_index, val_index])
    test_index = np.concatenate([test_index, test_index, test_index, test_index, test_index])
print(len(train_index))
print(len(val_index))
print(len(test_index))

polyak_decay = args.polyak
params = json.load(open(args.config, 'r'))
json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
mae = MAE.from_params(params).to(device)
# initialize
init_batch_size = 256
init_index = np.random.choice(train_index, init_batch_size, replace=False)
init_data, _ = get_batch(train_data, init_index)
init_data = init_data.to(device)
mae.eval()
mae.initialize(init_data, init_scale=1.0)
# create shadow mae for ema
params = json.load(open(args.config, 'r'))
mae_shadow = MAE.from_params(params).to(device)
exponentialMovingAverage(mae, mae_shadow, polyak_decay, init=True)
print(args)


def get_optimizer(learning_rate, parameters):
    if opt == 'adam':
        return optim.Adam(parameters, lr=learning_rate, betas=betas, eps=eps)
    elif opt == 'adamax':
        return optim.Adamax(parameters, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError('unknown optimization method: %s' % opt)


opt = args.opt
betas = (0.95, polyak_decay)
eps = 1e-8

if opt == 'adam':
    lr = 1e-3
elif opt == 'adamax':
    lr = 2e-3
else:
    raise ValueError('unknown optimization method: %s' % opt)

optimizer = get_optimizer(lr, mae.parameters())
step_decay = 0.999995
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)
lr_min = lr / 20

decay_rate = 0.5
max_decay = 3
schedule = args.schedule
decay = 0
patient = 0


def train(epoch):
    print('Epoch: %d (lr=%.6f (%s), patient=%d (%d), decay=%d (%d))' % (epoch, lr, opt, patient, schedule, decay, max_decay))
    mae.train()
    recon_loss = 0
    kl_loss = 0
    pkl_mean = 0
    pkl_mean_loss = 0
    pkl_std = 0
    pkl_std_loss = 0
    num_insts = 0
    num_batches = 0

    num_back = 0
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(iterate_minibatches(train_data, train_index, args.batch_size, True)):
        data = data.to(device)

        batch_size = len(data)
        optimizer.zero_grad()
        loss, recon, kl, pkl_m, pkl_s, loss_pkl_mean, loss_pkl_std = mae.loss(data, nsamples=training_k, eta=eta, gamma=gamma, free_bits=free_bits)
        loss.backward()
        clip_grad_norm_(mae.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        exponentialMovingAverage(mae, mae_shadow, polyak_decay)

        with torch.no_grad():
            num_insts += batch_size
            num_batches += 1
            recon_loss += recon.sum()
            kl_loss += kl.sum()
            pkl_mean += pkl_m
            pkl_std += pkl_s
            pkl_mean_loss += loss_pkl_mean
            pkl_std_loss += loss_pkl_std

        if batch_idx % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            train_loss = recon_loss / num_insts + kl_loss / num_insts + pkl_mean_loss / num_batches + pkl_std_loss / num_batches
            log_info = '[{}/{} ({:.0f}%)] Loss: {:.2f} (recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f})'.format(
                batch_idx * batch_size, len(train_index), 100. * num_insts / len(train_index),
                train_loss, recon_loss / num_insts, kl_loss / num_insts,
                pkl_mean / num_batches, pkl_std / num_batches,
                pkl_mean_loss / num_batches, pkl_std_loss / num_batches)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    train_loss = recon_loss / num_insts + kl_loss / num_insts + pkl_mean_loss / num_batches + pkl_std_loss / num_batches
    print('Average loss: {:.2f} (recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f}), time: {:.1f}s'.format(
        train_loss, recon_loss / num_insts, kl_loss / num_insts,
                    pkl_mean / num_batches, pkl_std / num_batches,
                    pkl_mean_loss / num_batches, pkl_std_loss / num_batches,
                    time.time() - start_time))


def eval(eval_data, eval_index):
    mae_shadow.eval()
    recon_loss = 0.
    kl_loss = 0.
    pkl_mean = 0.
    pkl_mean_loss = 0.
    pkl_std_loss = 0.
    pkl_std = 0.
    num_insts = 0
    num_batches = 0
    for i, (data, _) in enumerate(iterate_minibatches(eval_data, eval_index, 100, False)):
        data = data.to(device)

        batch_size = len(data)
        loss, recon, kl, pkl_m, pkl_s, loss_pkl_mean, loss_pkl_std = mae_shadow.loss(data, nsamples=test_k, eta=eta, gamma=gamma)

        num_insts += batch_size
        num_batches += 1
        recon_loss += recon.sum()
        kl_loss += kl.sum()
        pkl_mean += pkl_m
        pkl_std += pkl_s
        pkl_mean_loss += loss_pkl_mean
        pkl_std_loss += loss_pkl_std

    recon_loss /= num_insts
    kl_loss /= num_insts
    pkl_mean /= num_batches
    pkl_mean_loss /= num_batches
    pkl_std /= num_batches
    pkl_std_loss /= num_batches
    test_loss = recon_loss + kl_loss + pkl_mean_loss + pkl_std_loss
    test_elbo = recon_loss + kl_loss
    bits_per_pixel = test_elbo / (nx * np.log(2.0))

    print('loss: {:.2f} (elbo: {:.2f} recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f}), BPD: {:.2f}'.format(
        test_loss, test_elbo, recon_loss, kl_loss,
        pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss,
        bits_per_pixel))
    return test_loss, recon_loss, kl_loss, pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss, bits_per_pixel


def reconstruct():
    mae_shadow.eval()
    n = 128
    data, _ = get_batch(test_data, test_index[:n])
    data = data.to(device)

    recon_img, _ = mae_shadow.reconstruct(data)
    comparison = torch.cat([data, recon_img], dim=0).cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(2)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct.png'
    save_image(comparison, os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))


def calc_nll():
    mae_shadow.eval()
    start_time = time.time()
    nll_elbo = 0.
    recon_err = 0.
    kl_err = 0.
    nll_iw = 0.
    num_insts = 0
    for i, (data, _) in enumerate(iterate_minibatches(test_data, test_index, 1, False)):
        data = data.to(device)

        batch_size = len(data)
        num_insts += batch_size
        (elbo, recon, kl), iw = mae_shadow.nll(data, k)
        nll_elbo += elbo.sum()
        recon_err += recon.sum()
        kl_err += kl.sum()
        nll_iw += iw.sum()

    nll_elbo /= num_insts
    recon_err /= num_insts
    kl_err /= num_insts
    nll_iw /= num_insts
    bits_per_pixel = nll_iw / (nx * np.log(2.0))
    print('Test NLL: ELBO: {:.2f} (recon: {:.2f}, kl: {:.2f}), IW: {:.2f}, BPD: {:.2f}, time: {:.2f}s'.format(nll_elbo, recon_err, kl_err, nll_iw, bits_per_pixel, (time.time() - start_time)))
    return (nll_elbo, recon_err, kl_err), nll_iw, bits_per_pixel


best_epoch = 0
best_loss = 1e12
best_elbo = 1e12
best_recon = 1e12
best_kl = 1e12
best_pkl_mean = 1e12
best_pkl_mean_loss = 1e12
best_pkl_std = 1e12
best_pkl_std_loss = 1e12
best_bpd = 1e12

for epoch in range(1, args.epochs + 1):
    train(epoch)
    lr = scheduler.get_lr()[0]
    print('----------------------------------------------------------------------------------------------------------------------------')
    with torch.no_grad():
        loss, recon, kl, pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss, bits_per_pixel = eval(train_data, val_index)
    elbo = recon + kl
    if elbo < best_elbo:
        patient = 0
        torch.save(mae_shadow.state_dict(), model_name)

        best_epoch = epoch
        best_loss = loss
        best_elbo = elbo
        best_recon = recon
        best_kl = kl
        best_pkl_mean = pkl_mean
        best_pkl_mean_loss = pkl_mean_loss
        best_pkl_std = pkl_std
        best_pkl_std_loss = pkl_std_loss
        best_bpd = bits_per_pixel
    elif patient >= schedule:
        lr = lr * decay_rate
        optimizer = get_optimizer(lr, mae.parameters())
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_decay)
        patient = 0
        decay +=1
    else:
        patient += 1

    print('Best: {:.2f} (elbo: {:.2f} recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f}), BPD: {:.2f}, epoch: {}'.format(
        best_loss, best_elbo, best_recon, best_kl,
        best_pkl_mean, best_pkl_std, best_pkl_mean_loss, best_pkl_std_loss,
        best_bpd, best_epoch))
    print('============================================================================================================================')

    if lr < lr_min or decay == max_decay:
        break

mae_shadow.load_state_dict(torch.load(model_name))
with torch.no_grad():
    reconstruct()
    sample_z, _ = mae_shadow.sample_from_proir(256, device=device)
    sample_x, _ = mae_shadow.decode(sample_z, random_sample=True)
    image_file = 'sample.png'
    save_image(sample_x.cpu(), os.path.join(result_path, image_file), nrow=16, normalize=True, scale_each=True, range=(-1, 1))

    print('Final val:')
    eval(train_data, val_index)
    print('Final test:')
    eval(test_data, test_index)
    print('----------------------------------------------------------------------------------------------------------------------------')
    calc_nll()
    print('============================================================================================================================')
