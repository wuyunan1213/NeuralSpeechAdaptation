import os
import random
import keras
# import tensorflow as tf
import numpy as np

import keras
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Add, Concatenate
from keras_lr_multiplier import LRMultiplier
from keras.callbacks import Callback
from keras.optimizers import SGD, Adam
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import Callback
from keras.callbacks import LambdaCallback, ModelCheckpoint
import pickle as pkl
import matplotlib.pyplot as plt

import scipy, time
import scipy.io
import scipy.stats

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

class Test_NBatchLogger(Callback):
    """
    A Logger that log  performance per batch.
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
        self.pred_l.extend(self.model.predict(self.test_l)[0])
        self.pred_h.extend(self.model.predict(self.test_h)[0])

def ff_nn_one(n_slow=40, n_fast = 10, n_inp = 30, lr_s = 1, lr_f = 10, activation = 'linear'):
    model = Sequential()

    inp = Input(shape = (n_inp,), name = 'input')
    slow = Dense(n_slow, activation=activation, name = 'slow')(inp)
    fast = Dense(n_fast, activation=activation, name = 'fast')(inp)
    s2 = Dense(n_slow, activation=activation, name = 's2')(slow)
    f2 = Dense(n_fast, activation=activation, name = 'f2')(fast)
    added = Concatenate()([s2, f2])
    ###use layers.concatenate rather than layers.add so that all the slow and fast units are only 
    ###concatenated but not added as the same shape. E.g., if we have 30 slow units and 30 fast units
    ###concatenate would give you a shape of 60 but add would give you a shape of 30. If we have, instead,
    ###30 slow units and 10 fast units using concatenate would give you a shape of 40 but add 
    ###would not work in this case.

    out = Dense(1, activation="sigmoid", name="output")(added)
    model = Model(inputs = inp, outputs = out)

    ####for now we will just have one fixed learning rate for the slow-fast pretraining, which
    #### is essentially a slow-slow model 
    model.compile(optimizer = LRMultiplier(Adam(lr=1e-3), {'slow':lr_s, 'fast':lr_f, 's2': lr_s, 'f2': lr_f}),
                #SGD(lr=0.05),
                loss='binary_crossentropy',
                metrics=['accuracy'])    
    return model

def slow_fast_nn_one(lr_s = 1, lr_f = 10, n_inp = 30, n_slow=40, n_fast=10, activation = 'linear', train_slow = False):
    inp = Input(shape = (n_inp,), name = 'input')
    slow = Dense(n_slow, activation=activation, name = 'slow')(inp)
    fast = Dense(n_fast, activation=activation, name = 'fast')(inp)
    #To be able to assign a different learning rate to the entire path in fast or slow
    s2 = Dense(n_slow, activation=activation, name = 's2')(slow)
    f2 = Dense(n_fast, activation=activation, name = 'f2')(fast)
    # added = Add()([slow, fast])
    added = Concatenate()([s2, f2])
    out = Dense(1, activation="sigmoid", name="output")(added)
    model = Model(inputs = inp, outputs = out)
    if(train_slow):
        ###haven't been able to use the LRMultiplier function yet
        model.compile(optimizer = LRMultiplier(Adam(lr=1e-3), {'slow':lr_s, 'fast':lr_f, 's2': lr_s, 'f2': lr_f}),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])        
    else:
        slow.trainable = False ##silence these to unfreeze the slow pathway 
        # s2.trainable = False
        model.compile(optimizer = SGD(lr=0.1),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def set_fs_weights_one(slow_model, lr_s = 1, lr_f = 10, activation = 'linear', train_slow = False):
    fs_model = slow_fast_nn_one(lr_s = lr_s, lr_f = lr_f, activation = activation, train_slow = train_slow)
    sm_weights = slow_model.get_weights()
    # print('SMWEIGHTS = ', sm_weights[4][10])
    fs_model.get_layer('slow').set_weights(sm_weights[:2])
    fs_model.get_layer('fast').set_weights(sm_weights[2:4])
    fs_model.get_layer('s2').set_weights(sm_weights[4:6])
    fs_model.get_layer('f2').set_weights(sm_weights[6:8])
    fs_model.get_layer('output').set_weights(sm_weights[8:10])
    # fs_model = clone_model(slow_model)
    # fs_weights = fs_model.get_weights()
    # print('fsWEIGHTS = ', fs_weights[4][10])
    # fs_model.get_layer('slow').set_weights(sm_weights[:2])
    # fs_model.get_layer('slow_2').set_weights(sm_weights[2:])
    # sm_weights = [6,] with (30,30)(30,0) (slow weights, bias), (30,30)(30,0)(fast weights, bias), (30,1)(1,)
    return fs_model

def test_d2_reliance(model, train_list, test_l, test_h, figType, lr_s = 1, lr_f = 10, batch_size = 40, epoch = 1):
    # model = clone_model(old_model)
    # ### SAVE WEIGHTS
    # model.compile(optimizer = LRMultiplier('adam', {'slow':lr_s, 'fast':lr_f}),
    #               #SGD(lr=0.1,),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    history = Test_NBatchLogger(test_l = test_l, test_h = test_h)
    fs_hist = model.fit(
        train_list[0], train_list[1],
        batch_size = batch_size,
        epochs = epoch,
        validation_data=(train_list[2], train_list[3]),
        callbacks = [history]
    )
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(history.acc)
    # ax1.plot(history.losses)

    # ax2.plot(history.pred_l)
    # ax2.plot(history.pred_h)

    # ax1.set_title('%s_Training_accuracy'%(figType))
    # ax1.set(xlabel = 'Batch', ylabel = 'Accuracy')
    # ax1.legend(['Accuracy', 'Loss'], loc = 'lower left')

    # ax2.set_title('%s_Test_Probability'%(figType))
    # ax2.set(xlabel = 'Batch', ylabel = 'Probability')
    # ax2.set_ylim((0, 1))
    # ax2.legend(['Low_F0', 'High_F0'], loc = 'lower left')

    # plt.tight_layout()

    # figname = train_fig_dir + '%s.png'%(figType)
    # plt.savefig(figname)
    # plt.close()
    test_l_pred = model.predict(test_l)
    test_h_pred = model.predict(test_h)

    return test_l_pred, test_h_pred, history.pred_l, history.pred_h


def mean_confidence_interval(data, confidence=0.95):
    ###assuming the input is an np array
    n = len(data)
    p = np.shape(data)[1]
    idx = int(p/2)
    low = data[:, :idx]
    high = data[:, idx:]
    m_l, se_l = np.mean(low, axis = 0), scipy.stats.sem(low)
    m_h, se_h = np.mean(high, axis = 0), scipy.stats.sem(high)
    #h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return (m_l, se_l, m_h, se_h)##change this to h if you want CIs.


def plot_mean_and_CI(mean, se, mean2 = None, se2 = None, color_shading=None, 
                    color_shading2=None):
    # plot the shaded range of the confidence intervals
    x = range(mean.shape[0])
    fig = plt.figure()
    plt.fill_between(range(mean.shape[0]), mean-se, mean+se,
                     color=color_shading, alpha=.5)
    plt.fill_between(range(mean2.shape[0]), mean2-se2, mean2+se2,
                 color=color_shading2, alpha=.5)
    plt.ylim((0,1))
    plt.legend(['LowF0', 'HighF0'], loc = 'lower right') 
    # plot the mean on top
    plt.plot(x, mean)
    plt.plot(x, mean2)
    
    return fig

def plot_exp_results(fname, lrr = 12, c1 = 'blue', c2 = 'orange'):
    r = []
    with open(fname, 'rb') as f:
        try:
            while True:
                r.append(pkl.load(f))
        except EOFError:
            pass
    r = np.array(r)
    rev_l_m, rev_l_se, rev_h_m, rev_h_se = mean_confidence_interval(r)
    f2 = plot_mean_and_CI(rev_l_m, rev_l_se, mean2 = rev_h_m, se2 = rev_h_se, color_shading = c1,
                                    color_shading2=c2)
    f2name = train_fig_dir+'%s_2_half_prob_%s.png'%(fname, lrr)
    f2.savefig(f2name)
