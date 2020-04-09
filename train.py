import os
import random
import keras
import tensorflow as tf
import numpy as np

from keras_lr_multiplier import LRMultiplier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import Callback
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
    opt = tf.keras.optimizers.Adam(lr = 0.002)#, decay = 1e-6)
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
        model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.05),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def set_fs_weights(slow_model, activation = 'linear', train_slow = True):
    fs_model = slow_fast_nn(activation = activation, train_slow = train_slow)
    sm_weights = slow_model.get_weights()
    fs_model.get_layer('slow').set_weights(sm_weights[:2])
    fs_model.get_layer('slow_2').set_weights(sm_weights[2:])
    return fs_model

class NBatchLogger(Callback):
    """
    A Logger that log average performance per batch.
    """
    def on_train_begin(self, logs = {}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

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


def test_d2_reliance(old_model, train_list, test_l, test_h, figType, batch_size = 1500, epoch = 1):
    model = tf.keras.models.clone_model(old_model)
    model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.05,),
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
    # plt.plot(history.acc)
    # plt.plot(history.losses)
    plt.plot(history.pred_l)
    plt.plot(history.pred_h)
    plt.title('%s_Test probabilities'%(figType))
    plt.ylabel('Prob')
    plt.xlabel('Batch')
    plt.ylim((0, 1))
    plt.legend(['Low_F0', 'High_F0'], loc = 'upper left')
    figname = train_fig_dir + '%s.png'%(figType)
    plt.savefig(figname)
    plt.close()
    test_l_pred = model.predict(test_l)
    test_h_pred = model.predict(test_h)

    return test_l_pred, test_h_pred
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
