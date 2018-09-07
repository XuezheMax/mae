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
parser.add_argument('--config', type=str, help='config file', required=True)
parser.add_argument('--data', choices=['mnist', 'omniglot'], help='data set', required=True)
parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=524287, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--eta', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--gamma', type=float, default=0.0, metavar='N', help='')
parser.add_argument('--schedule', type=int, default=20, help='schedule for learning rate decay')
parser.add_argument('--model_path', help='path for saving model file.', required=True)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda') if args.cuda else torch.device('cpu')

imageSize = 28
training_k = 1
test_k = 5
k = 4096
eta = args.eta
gamma = args.gamma

model_path = args.model_path
model_name = os.path.join(model_path, 'model.pt')
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = os.path.join(model_path, 'images')
if not os.path.exists(model_path):
    os.makedirs(model_path)


def iterate_minibatches(data, batch_size, shuffle, binarize):
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
    else:
        indices = None

    for start_idx in range(0, len(data), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if binarize:
            yield np.less_equal(np.random.random(data[excerpt].shape), data[excerpt]).astype(np.float32)
        else:
            yield data[excerpt]


dataset = args.data
train_data, test_data, n_val = load_datasets(dataset)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

val_data = train_data[-n_val:]
train_data = train_data[:-n_val]

val_binarized_data = np.concatenate([np.less_equal(np.random.random(val_data.shape), val_data).astype(np.float32) for _ in range(5)], axis=0)
test_binarized_data = np.concatenate([np.less_equal(np.random.random(test_data.shape), test_data).astype(np.float32) for _ in range(5)], axis=0)

np.random.shuffle(val_binarized_data)
np.random.shuffle(test_binarized_data)
print(train_data.shape)
print(val_binarized_data.shape)
print(test_binarized_data.shape)

params = json.load(open(args.config, 'r'))
json.dump(params, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
mae = MAE.from_params(params).to(device)
print(mae)

lr = 1e-3
optimizer = optim.Adam(mae.parameters(), lr=lr)
decay_rate = 0.5
schedule = args.schedule

patient = 0
decay = 0
max_decay = 6


def train(epoch):
    print('Epoch: %d lr=%.6f, decay rate=%.2f (schedule=%d, patient=%d, decay=%d)' % (epoch, lr, decay_rate, schedule, patient, decay))
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
    for batch_idx, binarized_data in enumerate(iterate_minibatches(train_data, args.batch_size, True, True)):
        binarized_data = torch.from_numpy(binarized_data).to(device).float()

        batch_size = len(binarized_data)
        optimizer.zero_grad()
        loss, recon, kl, pkl_m, pkl_s, loss_pkl_mean, loss_pkl_std = mae.loss(binarized_data, nsamples=training_k, eta=eta, gamma=gamma)
        loss.backward()
        clip_grad_norm_(mae.parameters(), 5.0)
        optimizer.step()

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
                batch_idx * batch_size, len(train_data), 100. * num_insts / len(train_data),
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
    print('Average loss: {:.2f} (recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f})'.format(
        train_loss, recon_loss / num_insts, kl_loss / num_insts,
        pkl_mean / num_batches, pkl_std / num_batches,
        pkl_mean_loss / num_batches, pkl_std_loss / num_batches))


def eval(eval_data):
    mae.eval()
    recon_loss = 0.
    kl_loss = 0.
    pkl_mean = 0.
    pkl_mean_loss = 0.
    pkl_std_loss = 0.
    pkl_std = 0.
    num_insts = 0
    num_batches = 0
    for i, binarized_data in enumerate(iterate_minibatches(eval_data, 512, False, False)):
        binarized_data = torch.from_numpy(binarized_data).to(device).float()

        batch_size = len(binarized_data)
        loss, recon, kl, pkl_m, pkl_s, loss_pkl_mean, loss_pkl_std = mae.loss(binarized_data, nsamples=test_k, eta=eta, gamma=gamma)

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

    print('loss: {:.2f} (elbo: {:.2f} recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f})'.format(
        test_loss, test_elbo, recon_loss, kl_loss,
        pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss))
    return test_loss, recon_loss, kl_loss, pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss


def reconstruct():
    mae.eval()
    n = 128
    data = torch.from_numpy(test_data[0:n]).to(device).float()
    binarized_data = torch.ge(data, 0.5).float()

    recon_img, recon_probs = mae.reconstruct(binarized_data)
    comparison = torch.cat([data, recon_probs, binarized_data, recon_img], dim=0).data.cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(4)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct.fixed.png'
    save_image(comparison, os.path.join(result_path, image_file), nrow=32)

    recon_img, recon_probs = mae.reconstruct(binarized_data, random_sample=True)
    comparison = torch.cat([data, recon_probs, binarized_data, recon_img], dim=0).data.cpu()
    reorder_index = torch.from_numpy(np.array([[i + j * n for j in range(4)] for i in range(n)])).view(-1)
    comparison = comparison[reorder_index]
    image_file = 'reconstruct.random.png'
    save_image(comparison, os.path.join(result_path, image_file), nrow=32)


def calc_nll():
    mae.eval()
    start_time = time.time()
    nll_elbo = 0.
    recon_err = 0.
    kl_err = 0.
    nll_iw = 0.
    num_insts = 0
    for i, binarized_data in enumerate(iterate_minibatches(test_binarized_data, 1, False, False)):
        binarized_data = torch.from_numpy(binarized_data).to(device).float()

        batch_size = len(binarized_data)
        num_insts += batch_size
        (elbo, recon, kl), iw = mae.nll_iw(binarized_data, k)
        nll_elbo += elbo.sum()
        recon_err += recon.sum()
        kl_err += kl.sum()
        nll_iw += iw.sum()

    nll_elbo /= num_insts
    recon_err /= num_insts
    kl_err /= num_insts
    nll_iw /= num_insts
    print('Test NLL: ELBO: {:.2f} (recon: {:.2f}, kl: {:.2f}), IW: {:.2f}, time: {:.2f}s'.format(nll_elbo, recon_err, kl_err, nll_iw, (time.time() - start_time)))
    return (nll_elbo, recon_err, kl_err), nll_iw


best_epoch = 0
best_loss = 1e12
best_elbo = 1e12
best_recon = 1e12
best_kl = 1e12
best_pkl_mean = 1e12
best_pkl_mean_loss = 1e12
best_pkl_std = 1e12
best_pkl_std_loss = 1e12

for epoch in range(1, args.epochs + 1):
    train(epoch)
    print('----------------------------------------------------------------------------------------------------------------------------')
    with torch.no_grad():
        loss, recon, kl, pkl_mean, pkl_std, pkl_mean_loss, pkl_std_loss = eval(val_binarized_data)
    elbo = recon + kl
    if elbo < best_elbo:
        patient = 0
        torch.save(mae.state_dict(), model_name)

        best_epoch = epoch
        best_loss = loss
        best_elbo = elbo
        best_recon = recon
        best_kl = kl
        best_pkl_mean = pkl_mean
        best_pkl_mean_loss = pkl_mean_loss
        best_pkl_std = pkl_std
        best_pkl_std_loss = pkl_std_loss
    elif patient >= schedule:
        mae.load_state_dict(torch.load(model_name))
        lr = lr * decay_rate
        optimizer = optim.Adam(mae.parameters(), lr=lr)
        patient = 0
        decay +=1
    else:
        patient += 1

    print('Best: {:.2f} (elbo: {:.2f} recon: {:.2f}, kl: {:.2f}, pkl (mean, std): {:.2f}, {:.2f}, pkl_loss (mean, std): {:.2f}, {:.2f}), epoch: {}'.format(
        best_loss, best_elbo, best_recon, best_kl,
        best_pkl_mean, best_pkl_std, best_pkl_mean_loss, best_pkl_std_loss,
        best_epoch))
    print('============================================================================================================================')

    if decay == max_decay:
        break

mae.load_state_dict(torch.load(model_name))
with torch.no_grad():
    reconstruct()
    sample_z = mae.sample_from_proir(400, device=device)
    sample_x, sample_probs = mae.decode(sample_z, random_sample=True)
    image_file = 'sample_binary.png'
    save_image(sample_x.data.cpu(), os.path.join(result_path, image_file), nrow=20)
    image_file = 'sample_cont.png'
    save_image(sample_probs.data.cpu(), os.path.join(result_path, image_file), nrow=20)

    print('Final val:')
    eval(val_binarized_data)
    print('Final test:')
    eval(test_binarized_data)
    print('----------------------------------------------------------------------------------------------------------------------------')
    calc_nll()
    print('============================================================================================================================')
