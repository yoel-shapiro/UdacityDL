# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:43:53 2016

@author: root
"""
import zipfile as _zf
import collections as _clct
import random as _rnd

import numpy as _np
import tensorflow as _tf
import matplotlib.pylab as _plt

_data_index = 0

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with _zf.ZipFile(filename) as f:
    data = _tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
  
def build_dataset(words, vocabulary_size):
  
  count = [['UNK', -1]]
  # n most common words, Counter (~dictionary) keys=words, values=counts
  count.extend(_clct.Counter(words).most_common(vocabulary_size - 1))

  # word IDs (=rank)
  # dictionary, key=word, value=rank (ordered by id\counts\rank)
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
    
  # integer (=rank, =id) representation of wikipedia text
  # replacing rare words with "UNK"
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word] # word id
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)  
    
  count[0][1] = unk_count

  # key=rank, value=word  
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  
  return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
  
  global _data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  
  batch = _np.ndarray(shape=(batch_size), dtype=_np.int32)
  labels = _np.ndarray(shape=(batch_size, 1), dtype=_np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  _buffer = _clct.deque(maxlen=span) # buffer() is a type of object
  
  for _ in range(span):
    _buffer.append(data[_data_index])
    _data_index = (_data_index + 1) % len(data)
    
  for i in range(batch_size // num_skips):
    
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    
    # generates a random order of all items in buffer
    for j in range(num_skips):     
      # keep until you hit one of the remaining unused items      
      while target in targets_to_avoid:
        target = _rnd.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = _buffer[skip_window]
      labels[i * num_skips + j, 0] = _buffer[target]
      
    # queue: adding next data item drops the oldest (due to maxlen limit)
    _buffer.append(data[_data_index])
    _data_index = (_data_index + 1) % len(data)
    
  return batch, labels
  

def cbow_batch(data, batch_size, half_window=4):
  
  global _data_index
  
  window = 2 * half_window
  
  _data_index = min(
    max(_data_index, half_window),
    len(data) - batch_size - half_window)
  
  labels = _np.array(
    data[_data_index:_data_index + batch_size], 
    dtype=_np.int32)
  
  batch = _np.ndarray(shape=(batch_size, window), dtype=_np.int32)
  for k in range(half_window):
    arm = half_window - k
    batch[:, k] = data[_data_index - arm:_data_index - arm + batch_size]
    batch[:, window - k - 1] = data[
      _data_index + arm:_data_index + arm + batch_size]
    
  return batch, labels
  
  
def plot_embedding(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  
  fig = _plt.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    _plt.scatter(x, y)
    _plt.annotate(
      label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
      ha='right', va='bottom')
  _plt.show()
  
  return fig
  
  
  
  
  
  
  
  