import os
import random
import keras
# import tensorflow as tf
import numpy as np

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import Callback
import keras
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Add
from keras_lr_multiplier import LRMultiplier
from keras.callbacks import Callback
from keras.optimizers import SGD, Adam

from keras.callbacks import LambdaCallback, ModelCheckpoint
import pickle as pkl
import matplotlib.pyplot as plt

import config
train_fig_dir = config.train_fig_dir
######################################################################################
###################################### Pre-training ##################################
######################################################################################
def load_data(name, folder = None):
    # if folder is None:
    #     file_name = "".join((os.getcwd(), "/", name))
    # else:
    file_name = folder + name
    print('LOADING DATA FROM', file_name)
    file = open(file_name, 'rb')
    data = pkl.load(file)
    file.close()
    return data

def loadOfInt(names, folder):
    if type(names) == str:
        return load_data(names, folder)
    all_dfs = []
    for n in names:
        df = load_data(n, folder)
        all_dfs.append(df)
    return all_dfs

def ff_nn(n_slow=30, activation = 'linear'):
    model = Sequential()
    model.add(Dense(n_slow, activation=activation, use_bias=True, name="slow"))
    model.add(Dense(1, activation='sigmoid', use_bias=True, name="slow_out"))
    opt = Adam(lr = 0.002)#, decay = 1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def slow_fast_nn(lr_s = 0.005, lr_f=0.05, n_inp = 30, n_slow=30, n_fast=30, activation = 'linear', train_slow = False):
    inp = Input(shape = (n_inp,), name = 'input')
    slow = Dense(n_slow, activation=activation, name = 'slow')(inp)
    fast = Dense(n_fast, activation=activation, name = 'fast')(inp)
    #To be able to assign a different learning rate to the entire path in fast or slow
    s2 = Dense(1, activation="linear", name = 'slow_2')(slow)
    f2 = Dense(1, activation="linear", name = 'fast_2')(fast)
    added = tf.keras.layers.Add()([s2, f2])
    out = Dense(1, activation="sigmoid", name="output")(added)
    model = Model(inputs = inp, outputs = out)
    if(train_slow):
        ###haven't been able to use the LRMultiplier function yet
        model.compile(optimizer = LRMultiplier('adam', {'slow':lr_s, 'fast':lr_f, 'slow_2':lr_s, 'fast_2':lr_f}),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])        
    else:
        slow.trainable = False ##silence these to unfreeze the slow pathway 
        s2.trainable = False
        model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def set_fs_weights(slow_model, activation = 'linear', train_slow = False):
    fs_model = slow_fast_nn(activation = activation, train_slow = train_slow)
    sm_weights = slow_model.get_weights()
    fs_model.get_layer('slow').set_weights(sm_weights[:2])
    fs_model.get_layer('slow_2').set_weights(sm_weights[2:])
    return fs_model

# class NBatchLogger(Callback):
#     """
#     A Logger that log average performance per batch.
#     """
#     def on_train_begin(self, logs = {}):
#         self.losses = []
#         self.acc = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('acc'))

class Test_NBatchLogger(Callback):
    """
    A Logger that log average performance per batch.
    """
    def __init__(self, test_l, test_h):
        self.test_l = test_l
        self.test_h = test_h

    def on_train_begin(self, logs = {}):
        self.losses = []
        self.acc = []
        self.pred_l = []
        self.pred_h = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.pred_l.extend(self.model.predict(self.test_l))
        self.pred_h.extend(self.model.predict(self.test_h))


def test_d2_reliance(old_model, train_list, test_l, test_h, figType, batch_size = 1500, lr_s = 1, lr_f = 10, epoch = 1):
    model = clone_model(old_model)
    model.compile(optimizer = LRMultiplier('adam', {'slow':lr_s, 'fast':lr_f, 'slow_2':lr_s, 'fast_2':lr_f}),
                  #SGD(lr=0.1,),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = Test_NBatchLogger(test_l = test_l, test_h = test_h)
    fs_hist = model.fit(
        train_list[0], train_list[1],
        batch_size = batch_size,
        epochs = epoch,
        validation_data=(train_list[2], train_list[3]),
        callbacks = [history]
    )
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(history.acc)
    ax1.plot(history.losses)

    ax2.plot(history.pred_l)
    ax2.plot(history.pred_h)

    ax1.set_title('%s_Training_accuracy'%(figType))
    ax1.set(xlabel = 'Batch', ylabel = 'Accuracy')
    ax1.legend(['Accuracy', 'Loss'], loc = 'lower left')

    ax2.set_title('%s_Test_Probability'%(figType))
    ax2.set(xlabel = 'Batch', ylabel = 'Probability')
    ax2.set_ylim((0, 1))
    ax2.legend(['Low_F0', 'High_F0'], loc = 'lower left')

    plt.tight_layout()

    figname = train_fig_dir + '%s.png'%(figType)
    plt.savefig(figname)
    plt.close()
    test_l_pred = model.predict(test_l)
    test_h_pred = model.predict(test_h)

    return test_l_pred, test_h_pred


def mean_confidence_interval(data, confidence=0.95):
    ###assuming the input is an np array
    n = len(data)
    m, se = np.mean(data, axis = 0), scipy.stats.sem(data)
    #h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-se, m+se##change this to h if you want CIs.


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    x = np.subtract(range(mean.shape[0]), 200)
    plt.fill_between(np.subtract(range(mean.shape[0]), 200), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x, mean, color_mean)
    # preds_l = model.predict(test_l) >= 0.5
    # preds_h = model.predict(test_h) >= 0.5
    # return sum(preds_l)/len(preds_l), sum(preds_h)/len(preds_h), model

# class NBatchLogger(Callback):
#     """
#     A Logger that log average performance per `display` steps.
#     """
#     def __init__(self, display):
#         self.step = 0
#         self.display = display
#         self.metric_cache = {}

#     def on_batch_end(self, batch, logs={}):
#         self.step += 1
#         for k in self.params['metrics']:
#             if k in logs:
#                 self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
#         if self.step % self.display == 0:
#             metrics_log = ''
#             for (k, v) in self.metric_cache.items():
#                 val = v / self.display
#                 if abs(val) > 1e-3:
#                     metrics_log += ' - %s: %.4f' % (k, val)
#                 else:
#                     metrics_log += ' - %s: %.4e' % (k, val)
#             print('step: {}/{} ... {}'.format(self.step,
#                                           self.params['steps'],
#                                           metrics_log))
#             self.metric_cache.clear()
