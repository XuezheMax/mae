import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.utils
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image

from mae.data import load_datasets, get_batch, iterate_minibatches
from mae.modules import MAE

parser = argparse.ArgumentParser(description='MAE Binary Image Example')
parser.add_argument('--data', choices=['mnist', 'omniglot', 'cifar10', 'lsun'], help='data set', default="mnist", required=False)
parser.add_argument('--seed', type=int, default=65537, metavar='S', help='random seed (default: 524287)')
parser.add_argument('--model_path', help='path for saving model file.', required=True)
parser.add_argument('--n_labels', default=10, type=int)
parser.add_argument("--method", choices=["kmeans", "knn", "svm", "linear"], required=True)
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

train_index_full = train_index

np.random.shuffle(train_index)
val_index = train_index[-n_val:]
train_index = train_index[:-n_val]

test_index = np.arange(len(test_data))
np.random.shuffle(test_index)

mae = MAE.load(model_path, device)

mode = args.mode


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


def kmeans(train_data, test_data, test_label, n_labels):
    # train_data / test_data: n_samples, dim
    alg = KMeans(n_clusters=n_labels, init='k-means++', random_state = 1,
                 n_init = 10, max_iter = 300, tol = 0.0001,
                 precompute_distances = 'auto', verbose = 0, copy_x = True, n_jobs = 10, algorithm ='auto')
    model = alg.fit(train_data)
    prediction = model.predict(test_data)
    acc = np.equal(prediction, test_label).sum() * 1.0 / len(prediction)
    print(f"Accuracy on test data is {acc}")


def knn(train_data, train_label, test_data, test_label, n_neighbors=10):
    # weights can be 'uniform'
    alg = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=10)
    model = alg.fit(train_data, train_label)
    acc = model.score(test_data, test_label)
    print(f"Accuracy on test data is {acc}")


def svm(train_data, train_label, test_data, test_label):
    clf = SVC(kernel='linear')
    clf.fit(train_data, train_label)
    acc = clf.score(test_data, test_label)
    print(f"Accuracy on test data is {acc}")


def data_iterator(data, label, batch_size):
    n_samples = data.size(0)
    num_batches = int(np.ceil(data.size(0) * 1.0 / batch_size))
    batches = []

    for i in range(num_batches):
        cur_batch_size = batch_size if i < num_batches - 1 else n_samples - batch_size * i
        batches.append((data[i * batch_size: i * batch_size + cur_batch_size], label[i * batch_size: i * batch_size + cur_batch_size]))

    np.random.shuffle(batches)
    for batch in batches:
        yield batch


def linear_classifier(train_data, train_label, val_data, val_label, test_data, test_label, n_labels, bs):
    n_sample, n_in = train_data.size(0), train_data.size(1)
    model = nn.Linear(n_in, n_labels)
    lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr)
    cross_entropy_loss = nn.CrossEntropyLoss(size_average=False, reduce=True)
    val_iter = 500
    steps = best_acc = bad_counter = decay_num = 0
    patience = 3
    best_params = None
    while True:
        for batch_idx, (data, labels) in enumerate(data_iterator(train_data, train_label, bs)):
            data = data.to(device)
            batch_size = len(data)

            logits = model(data)
            loss = cross_entropy_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % val_iter == 0:
                model.eval()
                with torch.no_grad():
                    predictions = torch.argmax(F.softmax(model(val_data), dim=1), dim=1)
                    acc = torch.mean(torch.equal(predictions, val_label)).item()
                    print(f"Evaluation acc on dev set = {acc}")
                    if acc > best_acc:
                        best_acc = acc
                        best_params = model.state_dict()
                        bad_counter = 0
                    else:
                        bad_counter += 1
                        if bad_counter >= 3:
                            lr *= 0.5
                            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                            decay_num += 1
                    if decay_num > patience:
                        model.load_state_dict(best_params)
                        test_pred = torch.argmax(F.softmax(model(test_data), dim=1), dim=1)
                        acc = torch.mean(torch.equal(test_pred, test_label)).item()
                        print(f"Evaluation acc on test set = {acc}")
                        exit(0)


def classify():
    time_start = time.time()

    method = args.method
    print('encoding:')
    if method == "linear":
        latent_codes_train, labels_train = encode(train_data, train_index)
        latent_codes_val, labels_val = encode(train_data, val_index)
        latent_codes_test, labels_test = encode(test_data, test_index)
    else:
        latent_codes_train, labels_train = encode(train_data, train_index_full)
        latent_codes_test, labels_test = encode(test_data, test_index)
        latent_codes_train, latent_codes_test = latent_codes_train.cpu().numpy(), latent_codes_test.cpu().numpy()
        labels_train, labels_test = labels_train.cpu().numpy(), labels_test.cpu().numpy()

    print('time: {:.1f}s'.format(time.time() - time_start))

    n_labels = args.n_labels
    n_neighbors =10

    if method == "kmeans":
        kmeans(latent_codes_train, latent_codes_test, labels_test, n_labels)
    elif method == "knn":
        knn(latent_codes_train, labels_train, latent_codes_test, labels_test, n_neighbors)
    elif method == "svm":
        svm(latent_codes_train, labels_train, latent_codes_test, labels_test)
    elif method == "linear":
        linear_classifier(latent_codes_train, labels_train, latent_codes_val, labels_val, latent_codes_test, labels_test, n_labels, 128)
