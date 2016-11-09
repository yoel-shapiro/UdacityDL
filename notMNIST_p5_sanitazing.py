# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import re
import IPython

ip = IPython.get_ipython()
ip.enable_pylab()

#folder = r'C:\Users\YoelS\Desktop\Udacity'
#f = open(os.path.join(folder, 'notMNIST.pickle'), 'rb')
#data = pickle.load(f)

image_size = 28

key = 'train_dataset'  # 'valid_dataset'  # 'test_dataset'  # 'train_dataset'
n_set = len(data[key]) - 1
n_partial = 3
index_duplicates = np.int32(np.array([]))
for k in np.arange(n_set - 1, 1, -1):

    if any(k == index_duplicates):
        continue

    if k % 1000 == 0:
        index_duplicates = np.unique(index_duplicates)
        print('found {} duplicates, {} images left to go'.format(
            str(len(index_duplicates)), str(k)))

    partial_diffs = np.zeros(k - 1)
    for r in range(n_partial):
        for c in range(n_partial):
            partial_diffs += np.abs(np.subtract(
                data[key][: k - 1, r, c], data[key][k, r, c]))

    index_candidates = np.argwhere(partial_diffs == 0).flatten()
    n_cand = len(index_candidates)
    if n_cand == 0:
        continue

    partial_diffs = partial_diffs[index_candidates]
    for r in range(n_partial, 2*n_partial + 2):
        for c in range(n_partial, 2*n_partial + 2):
            partial_diffs += np.abs(np.subtract(
                data[key][index_candidates, r, c], data[key][k, r, c]))

    index_cand2 = np.argwhere(partial_diffs == 0).flatten()
    n_cand = len(index_cand2)
    if n_cand == 0:
        continue

    partial_diffs = partial_diffs[index_cand2]
    index_candidates = index_candidates[index_cand2]
    for r in range(2*n_partial + 2, 3*n_partial + 4):
        for c in range(2*n_partial + 2, 3*n_partial + 4):
            partial_diffs += np.abs(np.subtract(
                data[key][index_candidates, r, c], data[key][k, r, c]))

    index_cand2 = np.argwhere(partial_diffs == 0).flatten()
    n_cand = len(index_cand2)
    if n_cand == 0:
        continue

    full_diffs = np.zeros(n_cand)
    index_candidates = index_candidates[index_cand2]
    for r in range(image_size):
        for c in range(image_size):
            full_diffs += np.abs(np.subtract(
                data[key][index_candidates, r, c], data[key][k, r, c]))

    index_full_match = np.argwhere(full_diffs == 0).flatten()
    index_full_match = index_candidates[index_full_match]
    n_full_match = len(index_full_match)
    if n_full_match > 1:
        index_duplicates = np.append(
            index_duplicates, np.int32(index_full_match[1:]))
        print('marking {} copies'.format(n_full_match))
    if n_full_match > 0:
        index_duplicates = np.append(index_duplicates, np.int32(k))

print('found {} duplicates, out of {}'.format(
            str(len(index_duplicates)), str(n_set + 1)))

index_keep = np.arange(n_set + 1)
index_keep = np.delete(index_keep, index_duplicates)

data_sanitized[key] = data[key][index_keep, :, :]

key = key.replace('_dataset', '_labels')
data_sanitized[key] = data[key][index_keep]


## cross checking - yielded zero duplicates
#
#key0 = 'train_dataset'
#key1 = 'test_dataset'
#n0 = len(data_sanitized[key0])
#n1 = len(data_sanitized[key1])
#n_partial = 3
#index_duplicates = np.int32(np.array([]))
#for k1 in range(n1):
#
#    if k1 % 500 == 0:
#        index_duplicates = np.unique(index_duplicates)
#        print('found {} duplicates, {} images left to go'.format(
#            str(len(index_duplicates)), str(n1 - k1)))
#
#    partial_diffs = np.zeros(n0)
#    for r in range(n_partial):
#        for c in range(n_partial):
#            partial_diffs += np.abs(np.subtract(
#                data_sanitized[key0][:, r, c],
#                data_sanitized[key1][k1, r, c]))
#
#    index_candidates = np.argwhere(partial_diffs == 0).flatten()
#    n_cand = len(index_candidates)
#    if n_cand == 0:
#        continue
#
#    partial_diffs = partial_diffs[index_candidates]
#    for r in range(n_partial, 2*n_partial + 2):
#        for c in range(n_partial, 2*n_partial + 2):
#            partial_diffs += np.abs(np.subtract(
#                data_sanitized[key0][index_candidates, r, c],
#                data_sanitized[key1][k1, r, c]))
#
#    index_cand2 = np.argwhere(partial_diffs == 0).flatten()
#    n_cand = len(index_cand2)
#    if n_cand == 0:
#        continue
#
#    partial_diffs = partial_diffs[index_cand2]
#    index_candidates = index_candidates[index_cand2]
#    for r in range(2*n_partial + 2, 3*n_partial + 4):
#        for c in range(2*n_partial + 2, 3*n_partial + 4):
#            partial_diffs += np.abs(np.subtract(
#                data_sanitized[key0][index_candidates, r, c],
#                data_sanitized[key1][k1, r, c]))
#
#    index_cand2 = np.argwhere(partial_diffs == 0).flatten()
#    n_cand = len(index_cand2)
#    if n_cand == 0:
#        continue
#
#    full_diffs = np.zeros(n_cand)
#    index_candidates = index_candidates[index_cand2]
#    for r in range(image_size):
#        for c in range(image_size):
#            full_diffs += np.abs(np.subtract(
#                data_sanitized[key0][index_candidates, r, c],
#                data_sanitized[key1][k1, r, c]))
#
#    index_full_match = np.argwhere(full_diffs == 0).flatten()
#    index_full_match = index_candidates[index_full_match]
#    n_full_match = len(index_full_match)
#    if n_full_match > 0:
#        index_duplicates = np.append(
#            index_duplicates, np.int32(index_full_match[1:]))
#
#print('found {} duplicates, out of {}'.format(
#            str(len(index_duplicates)), str(n1)))
#
#index_keep = np.arange(n0)
#index_keep = np.delete(index_keep, index_duplicates)



