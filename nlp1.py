# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import math
import os
import random
import numpy as np
import tensorflow as tf
from six.moves import range
from six.moves import cPickle
from sklearn.manifold import TSNE

import matplotlib.pylab as plt

import nlp_utils as uts

plt.close('all')

skipgram = False
cbow = True

n_nearest_print = 50000

folder = '/home/yoel/Data/Dropbox/Udacity/'
filename = folder + 'data/text8.zip'
  
words = uts.read_data(filename)
print('Data size %d' % len(words))

vocabulary_size = 50000
data, count, dictionary, reverse_dictionary = uts.build_dataset(
  words, vocabulary_size)
  
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
  uts._data_index = 0
  batch, labels = uts.generate_batch(
    data, batch_size=8, num_skips=num_skips, skip_window=skip_window)
  print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
  print('    batch:', [reverse_dictionary[bi] for bi in batch])
  print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


# %% embedding graph

def word2id(words):
  ids = []
  for w in words:
    if w in dictionary.keys():
      ids.append(dictionary[w])
    else:
      ids.append('UNK')
  return np.array(ids, ndmin=2).T


def id2word(ids):
  words = []
  for i in ids:
    words.append(reverse_dictionary[i])
  return words
  
  
if skipgram:
  
  batch_size = 128
  embedding_size = 128 # Dimension of the embedding vector.
  skip_window = 1 # How many words to consider left and right.
  num_skips = 2 # How many times to reuse an input to generate a label.
  # We pick a random validation set to sample nearest neighbors. here we limit 
  # the validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent. 
  valid_size = 16 # Random set of words to evaluate similarity on.
  valid_window = 100 # Only pick dev samples in the head of the distribution.
  valid_examples = np.array(random.sample(range(valid_window), valid_size))
  num_sampled = 64 # Number of negative examples to sample.
    
  graph = tf.Graph()
  
  n_analogy = 8
  
  with graph.as_default(): # , tf.device('/cpu:0'):
  
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    analogy_set = tf.placeholder(tf.int32, shape=[3, 1])
        
    # Variables.
    embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      
    softmax_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                           stddev=1.0 / math.sqrt(embedding_size)))
                           
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
            
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                 train_labels, num_sampled, vocabulary_size))
  
    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities 
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
      
    similarity = tf.matmul(
      valid_embeddings, tf.transpose(normalized_embeddings))
  
    embed_analogy = tf.nn.embedding_lookup(normalized_embeddings, analogy_set)
    
    analogy_vector = tf.sub(
      tf.add(embed_analogy[2, :, :], embed_analogy[1, :, :]),
      embed_analogy[0, :, :])
          
    analogy_vector /= tf.sqrt(tf.reduce_sum(tf.square(analogy_vector)))
    
    analogy_scores = tf.matmul(
      normalized_embeddings, tf.transpose(analogy_vector))
    
  num_steps = 100000 + 1
  uts._data_index = 0
  
  with tf.Session(graph=graph) as session:
    
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    
    for step in range(num_steps):
      
      batch_data, batch_labels = uts.generate_batch(
        data, batch_size, num_skips, skip_window)
        
      feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
      _, l = session.run([optimizer, loss], feed_dict=feed_dict)
      
      average_loss += l
      
      if step % 2000 == 0:
        if step > 0:
          average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step %d: %f' % (step, average_loss))
        average_loss = 0
        
      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % n_nearest_print == 0:
        sim = similarity.eval()
        for i in range(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8 # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k+1]
          log = 'Nearest to %s:' % valid_word
          for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log = '%s %s,' % (log, close_word)
          print(log)
          
    final_embeddings = normalized_embeddings.eval()    
      
    f_sg = folder + 'data/embeding_skipgram.pickle'
    with open(f_sg, 'wb') as f:
      cPickle.dump(final_embeddings, f, cPickle.HIGHEST_PROTOCOL) 

    plt.figure('Loss over time')
    plt.plot(step_num, loss_history)
    
    print('\nAnalogies')
    examples = [
      ['france', 'paris', 'italy'],
      ['man', 'king', 'woman'],
      ['dad', 'mom', 'boy'],
      ['cat', 'kitten', 'dog'],
      ['good', 'bad', 'black'],
      ['person', 'people', 'sheep']]
      
    for words in examples:
        
      res = analogy_scores.eval({analogy_set: word2id(words)})
      top_k = 8 # number of nearest neighbors
      nearest = res.flatten().argsort()[-top_k:]
      
      print('{} to {} is like {} to: {}'.format(
        words[0], words[1], words[2], ', '.join(id2word(nearest))))
  
  num_points = 400
  
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
  words = [reverse_dictionary[i] for i in range(1, num_points+1)]
  fig = uts.plot_embedding(two_d_embeddings, words)
  plt.gca().set_title('Skip-Gram')
  f_sg = folder + 'data/embeding_skipgram.pig'
  with open(f_sg, 'wb') as f:
    cPickle.dump(fig, f, cPickle.HIGHEST_PROTOCOL)
    
  analogy_words = [
    'france', 'paris', 'italy', 'rome',
    'man', 'king', 'woman', 'queen',
    'dad', 'mom', 'boy', 'girl',
    'cat', 'kitten', 'dog', 'puppy',
    'good', 'bad', 'black', 'white',
    'person', 'people', 'sheep', 'herd']
  
  analogy_ids = [dictionary[w] for w in analogy_words]
  
  tsne2 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  analogy_2d = tsne2.fit_transform(final_embeddings[analogy_ids])
  uts.plot_embedding(analogy_2d, analogy_words)
  
#  f_skipgram = folder + 'data/embeding_skipgram.pickle'
#  with open(f_skipgram, 'wb') as f:
#    cPickle.dump(final_embeddings, f, cPickle.HIGHEST_PROTOCOL)
  


# %% CBOW model

if cbow:
    
  uts._data_index = 0

  batch, labels = uts.cbow_batch(data, batch_size=8)

  print('\nbatch:')
  for k in range(len(labels)):
    print([reverse_dictionary[bi] for bi in batch[k,:]])
    
  print('\nlabels:')
  print([reverse_dictionary[li] for li in labels])

  batch_size = 128
  embedding_size = 128 # Dimension of the embedding vector.
  
  valid_size = 16 # Random set of words to evaluate similarity on.
  valid_window = 100 # Only pick dev samples in the head of the distribution.
  valid_examples = np.array(
    random.sample(range(10, 10 + valid_window), valid_size))
  num_sampled = 64 # Number of negative examples to sample.
  
  half_window = 4
  window = 2 * half_window
  
  graph = tf.Graph()
  
  with graph.as_default(): # , tf.device('/cpu:0'):
  
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    analogy_set = tf.placeholder(tf.int32, shape=[3, 1])
    
    # Variables.
    embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                           stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset[:, 0])
    for k in range(1, window):
      embed += tf.nn.embedding_lookup(embeddings, train_dataset[:, k])
    embed /= window
    
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(
        softmax_weights, softmax_biases, embed, 
        train_labels, num_sampled, vocabulary_size))
  
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    
    embed_analogy = tf.nn.embedding_lookup(normalized_embeddings, analogy_set)
    
    analogy_vector = tf.sub(
      tf.add(embed_analogy[2, :, :], embed_analogy[1, :, :]),
      embed_analogy[0, :, :])
    
    analogy_vector /= tf.sqrt(tf.reduce_sum(tf.square(analogy_vector)))
    
    analogy_scores = tf.matmul(
      normalized_embeddings, tf.transpose(analogy_vector))
  
  num_steps = 100000 + 1
  uts._data_index = 0
  
  with tf.Session(graph=graph) as session:
    
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    loss_history = []
    step_num = []
    
    for step in range(num_steps):
      
      batch_data, batch_labels = uts.generate_batch(
        data, batch_size, num_skips, skip_window)
        
      feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
      
      _, l = session.run([optimizer, loss], feed_dict=feed_dict)
      
      average_loss += l
      
      if step % 2000 == 0:
        if step > 0:
          average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step %d: %f' % (step, average_loss))
        step_num.append(step)
        loss_history.append(average_loss)
        average_loss = 0
        
      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % n_nearest_print == 0:
        sim = similarity.eval()
        for i in range(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8 # number of nearest neighbors
          # position = id
          nearest = (-sim[i, :]).argsort()[1:top_k+1]
          log = 'Nearest to %s:' % valid_word
          for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log = '%s %s,' % (log, close_word)
          print(log)
          
    cbow_embeddings = normalized_embeddings.eval()
       
    f_cbow = folder + 'data/embeding_cbow.pickle'
    with open(f_cbow, 'wb') as f:
      cPickle.dump(cbow_embeddings, f, cPickle.HIGHEST_PROTOCOL)   
      
    plt.figure('Loss over time')
    plt.plot(step_num, loss_history)
  
    print('\nAnalogies')
    examples = [
      ['france', 'paris', 'england'],
      ['man', 'king', 'woman'],
      ['dad', 'mom', 'boy'],
      ['cat', 'kitten', 'dog'],
      ['good', 'bad', 'black'],
      ['person', 'people', 'sheep']]
      
    for words in examples:
        
      res = analogy_scores.eval({analogy_set: word2id(words)})
      top_k = 8 # number of nearest neighbors
      nearest = res.flatten().argsort()[-top_k:]
      
      print('{} to {} is like {} to: {}'.format(
        words[0], words[1], words[2], ', '.join(id2word(nearest))))
        
  num_points = 400
  
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  two_d_embeddings = tsne.fit_transform(cbow_embeddings[1:num_points+1, :])
  
  words = [reverse_dictionary[i] for i in range(1, num_points+1)]
  fig = uts.plot_embedding(two_d_embeddings, words)
  plt.gca().set_title('CBOW')
  f_cbow = folder + 'data/embeding_cbow.pig'
  with open(f_cbow, 'wb') as f:
    cPickle.dump(fig, f, cPickle.HIGHEST_PROTOCOL)
    
#  analogy_words = [
#    'france', 'paris', 'england', 'london',
#    'man', 'king', 'woman', 'queen',
#    'dad', 'mom', 'boy', 'girl',
#    'cat', 'kitten', 'dog', 'puppy',
#    'good', 'bad', 'black', 'white',
#    'person', 'people', 'sheep', 'herd']
#  
#  analogy_ids = [dictionary[w] for w in analogy_words]
#  
#  tsne2 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#  analogy_2d = tsne2.fit_transform(cbow_embeddings[analogy_ids])  
#     
#  uts.plot_embedding(analogy_2d, analogy_words)   
#
#  analogy_ids = np.reshape(analogy_ids, (6, 4))
#  for k in range(6):
#    
#    temp = (
#      cbow_embeddings[analogy_ids[k, 0], :] + 
#      cbow_embeddings[analogy_ids[k, 2], :] -
#      cbow_embeddings[analogy_ids[k, 1], :])
#    n_temp = np.sqrt(np.sum(np.square(temp)))
#    temp /= n_temp
#    
#    temp1 = (
#      cbow_embeddings[analogy_ids[k, 1], :] -
#      cbow_embeddings[analogy_ids[k, 0], :])
#    n_temp = np.sqrt(np.sum(np.square(temp1)))
#    temp1 /= n_temp
#    
#    temp2 = (
#      cbow_embeddings[analogy_ids[k, 3], :] -
#      cbow_embeddings[analogy_ids[k, 2], :])
#    n_temp = np.sqrt(np.sum(np.square(temp2)))
#    temp2 /= n_temp
#    
#    print('{:.2f} [deg]'.format(
#      180 / np.pi * np.arccos(np.sum(np.multiply(
#        temp, cbow_embeddings[analogy_ids[k, 3], :])))))
#    
#    print('{:.2f} [deg]'.format(
#      180 / np.pi * np.arccos(np.sum(np.multiply(temp1, temp2)))))
#    
#    dist01 = np.sqrt(np.sum(np.square(
#      cbow_embeddings[analogy_ids[k, 0], :] - 
#      cbow_embeddings[analogy_ids[k, 1], :])))
#    
#    dist23 = np.sqrt(np.sum(np.square(
#      cbow_embeddings[analogy_ids[k, 2], :] - 
#      cbow_embeddings[analogy_ids[k, 3], :])))    
#    
#    dist02 = np.sqrt(np.sum(np.square(
#      cbow_embeddings[analogy_ids[k, 0], :] - 
#      cbow_embeddings[analogy_ids[k, 2], :])))
#    
#    dist13 = np.sqrt(np.sum(np.square(
#      cbow_embeddings[analogy_ids[k, 1], :] - 
#      cbow_embeddings[analogy_ids[k, 3], :])))    
#    
#    print('01 {:.2f}, 23 {:.2f}, 02 {:.2f}, 13 {:.2f}'.format(
#      dist01, dist23, dist02, dist13))  