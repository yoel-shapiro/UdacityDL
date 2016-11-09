# -*- coding: utf-8 -*-
"""
Udacity, Deep Learning
https://classroom.udacity.com/courses/ud730/
"""

from __future__ import print_function
from six.moves import cPickle as pickle
from six.moves import range

import os
import IPython
import itertools
import dl_utils
import matplotlib as mpl
import matplotlib.pylab as plt

import numpy as np
import tensorflow as tf

# TODO: add code from A1

# %% Personal goals
"""
visualizations:
- loss [done]
- gradient norm, gradient consistency  # TODO:
confusion maps [done in A1]
loss function with margin  # TODO:
compare non\sanitized datasets
streamline with:  # TODO: add to dl_utils
- PCA + t-SNE [done in A1]
- feature map visualization [done in A1]
"""

# %% Assignemnt 1
"""
Topic: logistic regression with sklearn

tasks:
a) data donwloading, sanitizing
b) train LogReg classifier
"""

# %% Assignment 2
"""
topic: training

tasks:
a) train with GD - (simple) Gradient Descent *
b) train with SGD - Stochastic Gradient descent [batches] *
c) add a RELU layer **

* DNN implementation of LogReg [single fully-connected layer]
** DNN with 1 hidden layer
"""

# %% Assignment 3
"""
topic: regularization

a) add L2 regularization
b) demonstrate overfitting, using small batches [used 8, LogReg more prone]
c) add dropout on DNN, with small batches
d) imporve DNN perfromance with learning rate decay
"""

# %%  Assignemnt 4
"""
topic: CNN - Convolutional NN

a) run out-of-box implementation
b) Replace strides with max pooling operation
c) improve performance with dropout, LR decay, etc.
"""

# %% Setup

ip = IPython.get_ipython()
ip.run_line_magic('matplotlib', 'qt')

data_folder = '/home/yoel/Data/Dropbox/Udacity/data/'
n_labels = 10

class_tags = [chr(k) for k in np.arange(ord('A'), ord('J') + 1)]

class_markers = itertools.cycle(
  ('d', 's', '.', 'o', '>', '*', '<', '^', 'v', 'p'))

A2 = False
A3 = False
A4 = False
A4b = True

# %% Data handling

(train_dataset, train_labels,
 valid_dataset, valid_labels,
 test_dataset, test_labels) = dl_utils.load_datasets(
   os.path.join(data_folder, 'notMNIST_sanitized.pickle2'))

if A2 or A3:
  reformat = dl_utils.reformat_dnn
elif A4 or A4b:
  reformat = dl_utils.reformat_cnn

(train_dataset, train_labels) = reformat(
  train_dataset, train_labels, n_labels)

(valid_dataset, valid_labels) = reformat(
  valid_dataset, valid_labels, n_labels)

(test_dataset, test_labels) = reformat(
  test_dataset, test_labels, n_labels)

input_size = train_dataset.shape[1]

# %% (simple) Gradient Descent    # HINT:

if A2:

  train_subset = 10000

  graph = tf.Graph()
  with graph.as_default():

    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([input_size, n_labels]))
    biases = tf.Variable(tf.zeros([n_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    #
    # these lines define handles(?) to POTENTIAL operations
    # operation evaluation is invoked by:
    # - calling session.run(fetch_dict), including the desired handle in fetch_dict
    # - using handle.eval()
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


  n_steps = 801
  n_update = 100
  train_loss = np.zeros((n_steps, 2))
  train_acc = np.zeros((int(np.ceil(float(n_steps)/n_update)), 2))
  valid_acc = np.zeros_like(train_acc)

  with tf.Session(graph=graph) as session:

    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(n_steps):
      # Run the computations. We tell .run() that we want to run the optimizer,
      # and get [fetch] the loss value and the training predictions returned as numpy
      # arrays.
      _, l, predictions = session.run([optimizer, loss, train_prediction])

      train_loss[step, :] = [step, l]

      if (step % n_update == 0):
        print('Loss at step %d: %.3f' % (step, l))

        index = step // n_update

        t_acc = dl_utils.accuracy(predictions, train_labels[:train_subset, :])
        train_acc[index, :] = [step, t_acc]
        print('Training accuracy: %.1f%%' % t_acc)

        # Calling .eval() on valid_prediction is basically like calling run(), but
        # just to get that one numpy array. Note that it recomputes all its graph
        # dependencies.
        v_acc = dl_utils.accuracy(valid_prediction.eval(), valid_labels)
        valid_acc[index, :] = [step, v_acc]
        print('Validation accuracy: %.1f%%' % v_acc)

    test_acc = dl_utils.accuracy(test_prediction.eval(), test_labels)
    print("Test accuracy: %.1f%%" % test_acc)


  fig_gd = dl_utils.plot_accuracy_loss(
    train_loss, train_acc, valid_acc, test_acc, 'GD')


# %% Stochastic Gradient Descent    # HINT:

if A2 or A3:

  batch_size = 8 # overfitting demo # 128
  beta_w = 1e-3

  graph = []
  del graph

  graph = tf.Graph()
  with graph.as_default():
    # STRUCTURE:
    # input layer = input_size, 784
    # output layer = n_labels, 10 [soft max]
    # no hidden layer, this is actually a linear classifier [matmul]

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # tf.Variable is persistent (vs. tf.placeholder)
    weights = tf.Variable(tf.truncated_normal([input_size, n_labels]))
    biases = tf.Variable(tf.zeros([n_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) +
      beta_w * tf.nn.l2_loss(weights))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


  n_steps = 3001
  n_update = 500
  train_loss = np.zeros((n_steps, 2))
  train_acc = np.zeros((int(np.ceil(float(n_steps)/n_update)), 2))
  valid_acc = np.zeros_like(train_acc)

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")

    for step in range(n_steps):

      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      train_loss[step, :] = [step, l]

      if (step % n_update == 0):

        print("Minibatch loss at step %d: %.3f" % (step, l))

        index = step // n_update

        t_acc = dl_utils.accuracy(predictions, batch_labels)
        train_acc[index, :] = [step, t_acc]
        print('Minibatch accuracy: %.1f%%' % t_acc)

        v_acc = dl_utils.accuracy(valid_prediction.eval(), valid_labels)
        valid_acc[index, :] = [step, v_acc]
        print('Validation accuracy: %.1f%%' % v_acc)

    test_acc = dl_utils.accuracy(test_prediction.eval(), test_labels)
    print("Test accuracy: %.1f%%" % test_acc)

  fig_sgd = dl_utils.plot_accuracy_loss(
    train_loss, train_acc, valid_acc, test_acc, 'SGD')


# %% DNN | 1 hidden layer    # HINT:

if A2 or A3:

  batch_size = 128 # overfitting demo : 8
  # supposed to be 1024, reducing for GPU protection (power supply)
  n_hidden = 256 # 128, 256 OK
  learning_rate = 0.005
  beta_w_in = 1e-2
  beta_w_out = 1e-3

  # learning decay
  decay_steps = 200
  decay_rate = 0.97

  graph = []
  del graph

  graph = tf.Graph()
  with graph.as_default():
    # STRUCTURE
    # input layer: input_size (-> n_hidden), linear combinations [matmul]
    # hidden layer: n_hidden (-> n_labels), ReLU
    # output layer: n_labels [soft max]


    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, input_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # learning rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    # tf.Variable is persistent (vs. tf.placeholder)
    # initial weights: Bengio suggests using U(-r, r),
    # where r ~ sqrt(6/(fan-in + fan-out))
    weights_in = tf.Variable(tf.truncated_normal(
      [input_size, n_hidden], stddev=2 / np.sqrt(input_size + n_hidden)))
    biases_in = tf.Variable(tf.zeros([n_hidden]))

    weights_out = tf.Variable(tf.truncated_normal(
      [n_hidden, n_labels], stddev=2 / np.sqrt(n_hidden + n_labels)))
    biases_out = tf.Variable(tf.zeros([n_labels]))

    # Training computation.
    h1 = tf.matmul(tf_train_dataset, weights_in) + biases_in

    # relu6 for overflow protection
    logits = tf.matmul(
      tf.nn.relu6(h1), weights_out) + biases_out
    # logits = tf.nn.relu_layer(h1, weights_out, biases_out)

  #   # applying dropout, excluding biases
  #  weights_out_dropout = tf.nn.dropout(weights_out, 0.5)
  #
  #  logits = tf.matmul(
  #    tf.nn.relu6(h1), weights_out_dropout) + biases_out

    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) +
      beta_w_in * tf.nn.l2_loss(weights_in) +
      beta_w_out * tf.nn.l2_loss(weights_out))

    # Optimizer.
    learning_rate_tag = tf.train.exponential_decay(
      learning_rate, global_step, decay_steps, decay_rate)

    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate_tag).minimize(
        loss, global_step=global_step) # optimizer updates global_step counter

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    valid_prediction = tf.nn.softmax(
      tf.matmul(
        tf.nn.relu6(
          tf.matmul(tf_valid_dataset, weights_in) + biases_in),
        weights_out) + biases_out)

    test_prediction = tf.nn.softmax(
      tf.matmul(
        tf.nn.relu6(
          tf.matmul(tf_test_dataset, weights_in) + biases_in),
        weights_out) + biases_out)

  n_steps = 3001
  n_update = 500
  train_loss = np.zeros((n_steps, 2))
  train_acc = np.zeros((int(np.ceil(float(n_steps)/n_update)), 2))
  valid_acc = np.zeros_like(train_acc)
  config = tf.ConfigProto(log_device_placement=True)

  with tf.Session(graph=graph, config=config) as session:
    tf.initialize_all_variables().run()
    print("Initialized")

    for step in range(n_steps):

      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      train_loss[step, :] = [step, l]

      if (step % n_update == 0):

        print("Minibatch loss at step %d: %.3f" % (step, l))

        index = step // n_update

        t_acc = dl_utils.accuracy(predictions, batch_labels)
        train_acc[index, :] = [step, t_acc]
        print('Minibatch accuracy: %.1f%%' % t_acc)

        v_acc = dl_utils.accuracy(valid_prediction.eval(), valid_labels)
        valid_acc[index, :] = [step, v_acc]
        print('Validation accuracy: %.1f%%' % v_acc)

    test_acc = dl_utils.accuracy(test_prediction.eval(), test_labels)
    print("Test accuracy: %.1f%%" % test_acc)

  fig_relu = dl_utils.plot_accuracy_loss(
    train_loss, train_acc, valid_acc, test_acc, 'ReLU')


# %% CNN - Convolutional Neural Network

if A4:

#  with tf.device('/gpu:0'):

  n_channels = 1 # grayscale
  n_hidden = 64 # 64

  image_size = 28
  batch_size = 16 * 4
  patch_size = 5
  depth = 16 * 2 # number of kernels

  relu_eps = 1e-3
  learning_rate = 0.5 # 0.001

  # learning decay
  decay_steps = 200
  decay_rate = 0.97 + 0.02

  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(batch_size, image_size, image_size, n_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # learning rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    # conv + ReLU, #kernels = depth
    l1_w_shape = [patch_size, patch_size, n_channels, depth]
    # conv + ReLU, #kernels = depth
    l2_w_shape = [patch_size, patch_size, depth, depth]
    # FC + ReLU | 2 conv layers with stride =2  =>  4 = 2^2
    l3_w_shape = [image_size // 4 * image_size // 4 * depth, n_hidden]
    # FC (Fully Connected) classifier layer
    l4_w_shape = [n_hidden, n_labels]

    layer1_weights = tf.Variable(tf.truncated_normal(
        l1_w_shape,stddev=1 * dl_utils.initWstd(l1_w_shape, l2_w_shape)))
    layer1_biases = tf.Variable(tf.zeros([depth]) * relu_eps) # per kernel bias

    layer2_weights = tf.Variable(tf.truncated_normal(
        l2_w_shape, stddev=1 * dl_utils.initWstd(l2_w_shape, l3_w_shape)))
    layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal(
        l3_w_shape, stddev=1 * dl_utils.initWstd(l3_w_shape, l4_w_shape)))
    layer3_biases = tf.Variable(tf.constant(0.0, shape=[n_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal(
        l4_w_shape, stddev=1 * dl_utils.initWstd(l4_w_shape, n_labels)))
    layer4_biases = tf.Variable(tf.constant(0.0, shape=[n_labels]))

    # Model.
    def model(data):

      # image: 28 x 28 x 1

      conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
      hidden = tf.nn.relu6(conv + layer1_biases)

      # fm1: 14 x 14 x 16

      conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
      hidden = tf.nn.relu6(conv + layer2_biases)

      # fm2: 7 x 7 x 16

      shape = hidden.get_shape().as_list()
      reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu6(tf.matmul(reshape, layer3_weights) + layer3_biases)

      # fm3\vector: 64 x 1

      return tf.matmul(hidden, layer4_weights) + layer4_biases # 10 x 1

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    learning_rate_tag = tf.train.exponential_decay(
      learning_rate, global_step, decay_steps, decay_rate)

    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate_tag).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


  n_update = 50
  n_steps = 1001  # #batches to run

  train_loss = np.zeros((n_steps, 2))
  train_acc = np.zeros((int(np.ceil(float(n_steps)/n_update)), 2))
  valid_acc = np.zeros_like(train_acc)

  config=tf.ConfigProto(log_device_placement=True)
  with tf.Session(graph=graph, config=config) as session:

    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(n_steps):

      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      train_loss[step, :] = [step, l]

      if (step % n_update == 0):

        print('Minibatch loss at step %d: %.3f' % (step, l))

        index = step // n_update

        t_acc = dl_utils.accuracy(predictions, batch_labels)
        train_acc[index, :] = [step, t_acc]
        print('Minibatch accuracy: %.1f%%' % t_acc)

        v_acc = dl_utils.accuracy(valid_prediction.eval(), valid_labels)
        valid_acc[index, :] = [step, v_acc]
        print('Validation accuracy: %.1f%%' % v_acc)

    test_acc = dl_utils.accuracy(test_prediction.eval(), test_labels)
    print('Test accuracy: %.1f%%' % test_acc)

  fig_cnn = dl_utils.plot_accuracy_loss(
    train_loss, train_acc, valid_acc, test_acc, 'CNN')



# %% CNN - Convolutional Neural Network   # HINT:

if A4b:

#  with tf.device('/gpu:0'):

  weight_factor = 6
  train_factor = 10 # 10
  
  strides = [1, 2, 2, 1]
  no_strides = [1, 1, 1, 1]
  
  n_channels = 1 # grayscale
  n_hidden = 64 # 64

  image_size = 28
  batch_size = 16 * 4 # [4 for x-entropy]
  patch_size = 5 # 5 7 9 11
  depth = 16 * 2 # number of kernels

  relu_eps = 1e-3
  learning_rate = 0.1 # [0.1 for x-entropy loss] 0.5 0.1 0.05 0.001

  # learning decay
  decay_steps = 100 # [100 x-entropy]
  decay_rate = 0.97 + 0.00002

  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(
      tf.float32, shape=(batch_size, image_size, image_size, n_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, n_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # learning rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.

    # Variables.
    # conv + ReLU, #kernels = depth
    l1_w_shape = [patch_size, patch_size, n_channels, depth]
    # conv + ReLU, #kernels = depth
    l2_w_shape = [patch_size, patch_size, depth, depth]
    # FC + ReLU | 2 conv layers with stride =2  =>  4 = 2^2
    l3_w_shape = [image_size // 4 * image_size // 4 * depth, n_hidden]
    # FC (Fully Connected) classifier layer
    l4_w_shape = [n_hidden, n_labels]

    layer1_weights = tf.Variable(tf.truncated_normal(
        l1_w_shape,
        stddev=weight_factor * dl_utils.initWstd(l1_w_shape, l2_w_shape)))
    layer1_biases = tf.Variable(tf.zeros([depth]) * relu_eps) # per kernel bias

    layer2_weights = tf.Variable(tf.truncated_normal(
        l2_w_shape, 
        stddev=weight_factor * dl_utils.initWstd(l2_w_shape, l3_w_shape)))
    layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal(
        l3_w_shape, 
        stddev=weight_factor * dl_utils.initWstd(l3_w_shape, l4_w_shape)))
    layer3_biases = tf.Variable(tf.constant(0.0, shape=[n_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal(
        l4_w_shape, 
        stddev=weight_factor * dl_utils.initWstd(l4_w_shape, n_labels)))
    layer4_biases = tf.Variable(tf.constant(0.0, shape=[n_labels]))
    
    # Model.
    def model(data):

      # image: 28 x 28 x 1

      # full convolution (stride=1) + pooling
      conv = tf.nn.conv2d(data, layer1_weights, no_strides, padding='SAME')
      hidden = tf.nn.relu6(
        tf.nn.max_pool(conv, [1, 28, 28, 1], strides, 'SAME') + 
        layer1_biases)

      # fm1: 14 x 14 x 16

      conv = tf.nn.conv2d(hidden, layer2_weights, no_strides, padding='SAME')
      hidden = tf.nn.relu6(
        tf.nn.max_pool(conv, [1, 14, 14, 1], strides, 'SAME') + 
        layer2_biases)

      # fm2: 7 x 7 x 16

      shape = hidden.get_shape().as_list()
      reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu6(tf.matmul(reshape, layer3_weights) + layer3_biases)

      # fm3\vector: 64 x 1

      return tf.matmul(hidden, layer4_weights) + layer4_biases # 10 x 1

    # Training computation.
    logits = model(tf_train_dataset)
#    loss = tf.reduce_mean(
#      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(
      tf.contrib.losses.hinge_loss(logits, tf_train_labels))
      
    # Optimizer.
    learning_rate_tag = tf.train.exponential_decay(
      learning_rate, global_step, decay_steps, decay_rate)

    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate_tag).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


  n_update = 50 * train_factor
  n_steps = (1000 * train_factor) + 1 # #batches to run

  train_loss = np.zeros((n_steps, 2))
  train_acc = np.zeros((int(np.ceil(float(n_steps)/n_update)), 2))
  valid_acc = np.zeros_like(train_acc)

  config=tf.ConfigProto(log_device_placement=True)
  with tf.Session(graph=graph, config=config) as session:

    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(n_steps):

      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      train_loss[step, :] = [step, l]

      if (step % n_update == 0):

        print('Minibatch loss at step %d: %.3f' % (step, l))

        index = step // n_update

        t_acc = dl_utils.accuracy(predictions, batch_labels)
        train_acc[index, :] = [step, t_acc]
        print('Minibatch accuracy: %.1f%%' % t_acc)

        v_acc = dl_utils.accuracy(valid_prediction.eval(), valid_labels)
        valid_acc[index, :] = [step, v_acc]
        print('Validation accuracy: %.1f%%' % v_acc)

    test_acc = dl_utils.accuracy(test_prediction.eval(), test_labels)
    print('Test accuracy: %.1f%%' % test_acc)

  fig_cnn = dl_utils.plot_accuracy_loss(
    train_loss, train_acc, valid_acc, test_acc, 'CNN')


"""
... adding Dropout
OR BATCH NORMALIZATION
"""

plt.show()
