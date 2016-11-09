# -*- coding: utf-8 -*-
"""
Utilities for Udacity Deep Learning assignments

Created on Mon Aug 22 23:43:53 2016

@author: yoel
"""
from __future__ import print_function
import numpy as _np
from six.moves import cPickle as _pickle
import matplotlib as _mpl
import matplotlib.pylab as _plt


def load_datasets(filename):
  """load and unpack dictionary with train, validation, and test sets"""

  with open(filename, 'rb') as f:

    saved = _pickle.load(f)

    train_dataset = saved['train_dataset']
    train_labels = saved['train_labels']
    valid_dataset = saved['valid_dataset']
    valid_labels = saved['valid_labels']
    test_dataset = saved['test_dataset']
    test_labels = saved['test_labels']

    del saved  # hint to help gc free up memory

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

  return (train_dataset, train_labels,
          valid_dataset, valid_labels,
          test_dataset, test_labels)


def reformat_dnn(dataset, labels, n_labels):
  """data conversions for DNN
  typing: cast to float32
  input: flatten matrix (image) to vector
  labels: tags to 1-hot encoding (0/1 mask)"""

  if dataset.ndim == 3:
    dataset = dataset.reshape(
      (dataset.shape[0], dataset.shape[1] * dataset.shape[2]))

  dataset = dataset.astype(_np.float32)

  labels = (_np.arange(n_labels) == labels[:,None]).astype(_np.float32)

  print('data:', dataset.shape, 'labels:', labels.shape)

  return dataset, labels


def reformat_cnn(dataset, labels, n_labels):
  """data conversions for CNN
  typing: cast to float32
  labels: tags to 1-hot encoding (0/1 mask)"""
  if dataset.ndim == 3:

    image_size = dataset[0].shape[0]

    if dataset[0].ndim > 2:
      n_channels = dataset[0].shape[3]
    else:
      n_channels = 1

    dataset = dataset.reshape(
      (-1, image_size, image_size, n_channels))

  dataset = dataset.astype(_np.float32)

  labels = (_np.arange(n_labels) == labels[:,None]).astype(_np.float32)

  print('data:', dataset.shape, 'labels:', labels.shape)

  return dataset, labels


def accuracy(predictions, labels):
    # +1 where highest prediction matches label-mask
    return ( 100 *
      _np.sum(_np.argmax(predictions, 1) == _np.argmax(labels, 1))
      / predictions.shape[0])


def plot_accuracy_loss(train_loss, train_acc,
                       valid_acc=None, test_acc=None, fig_name=None):
  """plot training history, i.e. accuracy and loss"""

  legend = False
  _mpl.rcParams.update(_mpl.rcParamsDefault)
  _plt.style.use('ggplot')

  if fig_name is None:
    fig_name = 'Accuracy and Loss'

  fig, ax = _plt.subplots(num=fig_name, nrows=2)

  ax[0].set_title('Accuracy')
  ax[0].set_ylim([0, 100])
  ax[0].plot(train_acc[:, 0], train_acc[:, 1], marker='o', label='train')

  if valid_acc is not None:
    ax[0].plot(valid_acc[:, 0], valid_acc[:, 1], marker='s', label='valid')
    legend = True

  if test_acc is not None:
    ax[0].hlines(test_acc, train_acc[0], train_acc[-1],
      linestyles='--', label='test')
    legend = True

  if legend:
    ax[0].legend(loc='lower right')

  ax[1].set_title('Train Loss')
  ax[1].plot(train_loss[:, 0], train_loss[:, 1])

  return fig


def initWstd(input_shape, output_shape):
  """initiale weight STD, to avoid overflow"""

  fan_in = _np.prod(input_shape)
  fan_out = _np.prod(output_shape)

  return 2.0 / _np.sqrt(fan_in + fan_out)






