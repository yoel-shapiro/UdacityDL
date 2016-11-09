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

root = r'C:\Users\YoelS\Desktop\Udacity'


#folders = ['notMNIST_small', 'notMNIST_large']
#
#for fld in folders:
#
#    candidates = os.listdir(os.path.join(root, fld))
#
#    for cnd in candidates:
#        if re.search('.pickle', cnd) is not None:
#            try:
#                with open(os.path.join(root, fld, cnd), 'rb') as f:
#                    dataset = pickle.load(f)
#            except Exception as e:
#                print('Unable to read ', cnd, ':', e)
#
#            nimages = dataset.shape[0]
#            nrows = 5
#            ncols = 8
#            fig, axes = plt.subplots(
#                num=cnd[0] + ' ' + fld[-5:] + ' : ' +
#                str(nimages) + ' samples',
#                nrows=nrows, ncols=ncols)
#
#            for r in range(nrows):
#                for c in range(ncols):
#
#                    index = np.int64(nimages * np.random.rand(1)[0])
#                    axes[r, c].imshow(dataset[index, :, :], cmap='Greys')



files = ['notMNIST.pickle', 'notMNIST_sanitized.pickle']

for fl in files:

    try:
        with open(os.path.join(root, fl), 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print('Unable to read ', fl, ':', e)

    for ds in ['train', 'valid', 'test']:

        data = dataset[ds + '_dataset']

        nimages = data.shape[0]
        nrows = 5
        ncols = 8
        fig, axes = plt.subplots(
            num=fl + ' ' + ds + ' : ' +
            str(nimages) + ' samples',
            nrows=nrows, ncols=ncols)

        for r in range(nrows):
            for c in range(ncols):

                index = np.int64(nimages * np.random.rand(1)[0])
                axes[r, c].imshow(data[index, :, :], cmap='Greys')
