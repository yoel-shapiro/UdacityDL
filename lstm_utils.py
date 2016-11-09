# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:43:01 2016

@author: root
"""
import zipfile as _zf
import collections as _clct
import random as _rnd
import string as _str

import numpy as _np
import tensorflow as _tf
import matplotlib.pylab as _plt
  
_vocabulary_size = len(_str.ascii_lowercase) + 1 # [a-z] + ' '
_first_letter = ord(_str.ascii_lowercase[0])


def read_data(filename):
  f = _zf.ZipFile(filename)
  for name in f.namelist():
    return _tf.compat.as_str(f.read(name))
  f.close()
  

def char2id(char):
  if char in _str.ascii_lowercase:
    return ord(char) - _first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + _first_letter - 1)
  else:
    return ' '


class BatchGenerator(object):
  
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    # each batch is composed of X samples (at different cursors)
    self._cursor = [offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = _np.zeros(
      shape=(self._batch_size, _vocabulary_size), dtype=_np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches


def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in _np.argmax(probabilities, 1)]


def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  # unshuffles letters, so you see each cursor's workrange separatly
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s
  
  
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return _np.sum(_np.multiply(labels, -_np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of 
  normalized probabilities.
  """
  r = _rnd.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1


def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = _np.zeros(shape=[1, vocabulary_size], dtype=_np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p


def random_distribution():
  """Generate a random column of probabilities."""
  b = _np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/_np.sum(b, 1)[:,None]
