# -*- coding: utf-8 -*-
import os
import time
import IPython
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import itertools

ip = IPython.get_ipython()
ip.enable_pylab()

root = '/home/yoel/Data/Dropbox/Udacity/data'

f = open(os.path.join(root, 'notMNIST_sanitized.pickle'), 'rb')
data = pickle.load(f)
f.close()

train_score = []
test_score = []
train_time = []
test_time = []

x = data['test_dataset']
x_test = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
y_test = data['test_labels']
n_test = len(y_test)

train_sizes = [50, 100, 1000, 5000, len(data['train_dataset'])]

for n_train in train_sizes:

    x = data['train_dataset'][:n_train, :, :]
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    y = data['train_labels'][:n_train]

    lg = LogisticRegression(
        dual=x.shape[0] > x.shape[1],
        C=min(1, np.log(n_train) / 20),
        tol=0.001, n_jobs=4)

    t0 = time.clock()
    lg.fit(x, y)
    t1 = time.clock()
    train_time.append(t1 - t0)

    y_predict = lg.predict(x)
    train_score.append(np.sum(y_predict.__eq__(y)) / n_train)

    t0 = time.clock()
    y_predict = lg.predict(x_test)
    t1 = time.clock()
    test_time.append(t1 - t0)

    test_score.append(np.sum(y_predict.__eq__(y_test)) / n_test)

    print('finished {} train size'.format(n_train))

fig, ax = plt.subplots(nrows=2)
lw = 2
marker = 'o'

ax[0].semilogx(train_sizes, train_time, lw=lw, marker=marker, label='train')
ax[0].set_title('Time [sec]')

ax[1].semilogx(train_sizes, train_score, lw=lw, marker=marker, label='train')
ax[1].semilogx(train_sizes, test_score, lw=lw, marker=marker, label='test')
ax[1].set_title('Score')

for k in range(2):
    ax[k].legend()
    ax[k].set_xticks(train_sizes)
    ax[k].set_xticklabels(train_sizes)
    ax[k].grid(axis='y')

if n_train == len(data['train_dataset']):

    f = open(os.path.join(root, 'LogReg.plt'), 'wb')
    pickle.dump(fig, f)
    f.close()

    f = open(os.path.join(root, 'LogReg.obj'), 'wb')
    pickle.dump(lg, f)
    f.close()


# ========================================================

f = open(os.path.join(root, 'notMNIST_sanitized.pickle2'), 'rb')
data = pickle.load(f)
f.close()

f = open(os.path.join(root, 'LogReg.obj2'), 'rb')
lg = pickle.load(f)
f.close()

x = data['test_dataset']
x_test = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
y_test = data['test_labels']
y_predict = lg.predict(x_test)

classes = [chr(k) for k in np.arange(ord('A'), ord('J') + 1)]
n_class = len(classes)


# %%

ip.enable_pylab()
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('seaborn-muted')
# 'seaborn-muted'  'ggplot'

fig_p, ax_p = plt.subplots(
    num='Probabilities', nrows=n_class, ncols=n_class,
    sharex=True, sharey=True)

x_ticks = np.linspace(0, 1, 6)

for t in range(n_class):
    index_t = np.argwhere(y_test == t).flatten()
    prob = lg.predict_proba(x_test[index_t])

    for l in range(n_class):
        index_l = np.argwhere(y_predict[index_t] == l).flatten()
        n_obs = len(index_l)
        if n_obs > 8:
            ax_p[t, l].hist(
                prob[index_l, l], normed=True, bins=int(np.sqrt(n_obs)))
        else:
            ax_p[t, l].plot(prob[index_l, l], np.ones(n_obs), '.')

        ax_p[t, l].set_xticks(x_ticks)
        ax_p[t, l].set_xticklabels(np.round(x_ticks, decimals=1))


mixup = np.zeros((n_class, n_class))
for t in range(n_class):
    index = np.argwhere(y_test == t).flatten()
    for l in range(n_class):
        if l != t:
            mixup[t, l] = 100 * np.sum(y_predict[index] == l) / len(index)

fig2, ax2 = plt.subplots()
ax2.set_title('Mixup Rates')
cax = ax2.imshow(mixup, cmap='RdPu', interpolation='nearest')
cbar = fig2.colorbar(cax, format='%.2f')
ax2.set_xticks(range(n_class))
ax2.set_yticks(range(n_class))
ax2.set_xticklabels(classes)
ax2.set_yticklabels(classes)
ax2.set_xlabel('label [prediction]')
ax2.set_ylabel('tag [truth]')
ax2.grid('off')


fig_feat, ax_feat = plt.subplots(nrows=2, ncols=5)
for r in range(2):
    for c in range(5):
        vals = lg.coef_[c + 5 * r, :]
        vals = np.abs(vals)
        vals = (vals - min(vals)) / (max(vals) - min(vals))
        ax_feat[r, c].imshow(
            np.reshape(vals, (28, 28)),
            cmap='Greys', interpolation='nearest')
        ax_feat[r, c].set_title(classes[c + r * 5])

plt.show()


# %% Visualize manifolds
#
# use PCA for dimensionality reduction
# use t-SNE to visualize manifold

try:
    del data
except:
    print(' ')

# try w/o _sanitized
f = open(os.path.join(root, 'notMNIST_sanitized.pickle2'), 'rb')
data = pickle.load(f)
f.close()

x = np.reshape(data['train_dataset'], (len(data['train_dataset']), 28 * 28))

n_pca = 50
pca = PCA(n_components=n_pca)
pca.fit(x)
xt = pca.transform(x)

tsne = TSNE()
n_tsne = 2000  # memory limit
x_embedd = tsne.fit_transform(xt[:n_tsne, :])

fig_pca, ax_pca = plt.subplots(num='PCA')
ax_pca.plot(np.cumsum(pca.explained_variance_ratio_), lw=3)


mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')  # fivethirtyeight')


classes = [chr(k) for k in np.arange(ord('A'), ord('J') + 1)]
marker = itertools.cycle(('d', 's', '.', 'o', '>', '*', '<', '^', 'v', 'p'))
fig_mnfld, ax_mnfld = plt.subplots(num='Manifold Embedding with t-SNE')
for k_label in range(10):
    index = np.argwhere(data['train_labels'][:n_tsne] == k_label).flatten()
    ax_mnfld.plot(
        x_embedd[index, 0], x_embedd[index, 1],
        linestyle='None', marker=marker.next(), ms=10,
        label=classes[k_label])

ax_mnfld.legend()

plt.show()


# %% Visualize PCA components

plt.close('all')

fig_fm, ax_fm = plt.subplots(num='PCA Feature Maps', nrows=4, ncols=8)
for r in range(ax_fm.shape[0]):
    for c in range(ax_fm.shape[1]):
        counter = c + r * ax_fm.shape[1]
        vals = pca.components_[counter, :]
        vals -= np.min(vals)
        vals /= np.max(vals)
        vals = np.reshape(vals, (28, 28))
        ax_fm[r, c].imshow(vals, interpolation='nearest')
        ax_fm[r, c].grid('off')
        ax_fm[r, c].tick_params(
            axis='both', which='both',
            bottom='off', top='off', left='off', right='off',
            labelbottom='off', labeltop='off',
            labelleft='off', labelright='off')

fig_fm.tight_layout()
plt.show()
