######################################################################################
#### This experiment has the same network architecture as
#### SF_2_LAYER_FULL_RCR but the weights are penalized by 
#### regularizers
######################################################################################
import os
import random
from random import shuffle
import keras
# import tensorflow as tf
import numpy as np

import keras
from keras.models import Sequential, Model, clone_model
from keras.layers import Input, Dense, Add, Concatenate
from keras_lr_multiplier import LRMultiplier
from keras.callbacks import Callback
from keras.optimizers import SGD, Adam
from keras import regularizers


import datetime
from math import ceil
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import Callback
from keras.callbacks import LambdaCallback, ModelCheckpoint
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from LR_SGD import LR_SGD

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
        self.acc.append(logs.get('accuracy'))
        self.pred_l.extend(self.model.predict(self.test_l)[0])
        self.pred_h.extend(self.model.predict(self.test_h)[0])
        
class EpochLogger(Callback):
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

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.pred_l.extend(self.model.predict(self.test_l)[0])
        self.pred_h.extend(self.model.predict(self.test_h)[0])


def ff_nn_one(n_slow=40, n_fast = 40, n_inp = 30, lr_s = 1, lr_f = 10, penalty = 0.001, activation = 'linear'):
    model = Sequential()

    inp = Input(shape = (n_inp,), name = 'input')
    slow = Dense(n_slow, activation=activation, name = 'slow')(inp)
    fast = Dense(n_fast, kernel_regularizer=regularizers.l1(penalty), activation=activation, name = 'fast')(inp)
    s2 = Dense(n_slow, activation=activation, name = 's2')(slow)
    f2 = Dense(n_fast, kernel_regularizer=regularizers.l1(penalty), activation=activation, name = 'f2')(fast)
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
    model.compile(optimizer = LRMultiplier(Adam(lr=5e-5), {'slow':lr_s, 'fast':lr_f, 's2': lr_s, 'f2': lr_f}),
                #SGD(lr=0.05),
                loss='binary_crossentropy',
                metrics=['accuracy'])    
    return model

def slow_fast_nn_one(lr_s = 1, lr_f = 10, n_inp = 30, n_slow=40, n_fast=40, penalty = 0.001, activation = 'linear', train_slow = False):
    inp = Input(shape = (n_inp,), name = 'input')
    slow = Dense(n_slow, activation=activation, name = 'slow')(inp)
    fast = Dense(n_fast, kernel_regularizer=regularizers.l1(penalty), activation=activation, name = 'fast')(inp)
    #To be able to assign a different learning rate to the entire path in fast or slow
    s2 = Dense(n_slow, activation=activation, name = 's2')(slow)
    f2 = Dense(n_fast, kernel_regularizer=regularizers.l1(penalty), activation=activation, name = 'f2')(fast)
    # added = Add()([slow, fast])
    added = Concatenate()([s2, f2])
    out = Dense(1, activation="sigmoid", name="output")(added)
    model = Model(inputs = inp, outputs = out)
    if(train_slow):
        ###haven't been able to use the LRMultiplier function yet
        model.compile(optimizer = LRMultiplier(Adam(lr=5e-5), {'slow':lr_s, 'fast':lr_f, 's2': lr_s, 'f2': lr_f}),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])        
    else:
        slow.trainable = False ##silence these to unfreeze the slow pathway 
        # s2.trainable = False
        model.compile(optimizer = SGD(lr=0.1),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def set_fs_weights_one(slow_model, lr_s = 1, lr_f = 10, penalty = 0.001, activation = 'linear', train_slow = False):
    fs_model = slow_fast_nn_one(lr_s = lr_s, lr_f = lr_f, penalty = penalty, activation = activation, train_slow = train_slow)
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

def paired_shuffle_split(x, y, split_size):
    ## assuming the lenght of x and y are the same, that is, the number of data points
    ## n is the same for x and y even though they might have different dimensions
    shuffle_x = []
    shuffle_y = []
    idx = list(range(0, np.shape(x)[0]))
    shuffle(idx)
    for id in idx:
        shuffle_x.append(x[id])
        shuffle_y.append(y[id])
    ss_x = np.array_split(shuffle_x, split_size)
    ss_y = np.array_split(shuffle_y, split_size)
    return ss_x, ss_y
        
    
    
def test_d2_reliance(model, train, test_l, test_h, figType, lr_s = 1, lr_f = 10, batch_size = 40, epoch = 1):
    if (np.shape(train[0])[0]) % batch_size == 0:
        split_size = ceil((np.shape(train[0])[0])/batch_size)
        if split_size == 1: # batch size smaller than row
            raise "In this experiment, batch size shouldn't equal trainin sample size"
        else:
            pred_l = []
            pred_h = []
            acc = []
            loss = []
            history = EpochLogger(test_l, test_h)
            for e in range(epoch):
                train_input, train_labels = paired_shuffle_split(train[0], train[1], split_size)
                batch_list = list(range(0, split_size))
                np.random.shuffle(batch_list)
                for row in batch_list:
                    model.fit(train_input[row], train_labels[row], callbacks = [history])
                    acc.append(history.acc[0])
                    loss.append(history.losses[0])
                    pred_l.append(history.pred_l[0])
                    pred_h.append(history.pred_h[0])
                    # pred_l.append(model.predict(test_l)[0][0])
                    # pred_h.append(model.predict(test_h)[0][0])
    else:
        raise "Batch size not divisible by sample size"

    return pred_l[-1], pred_h[-1], pred_l, pred_h, acc, loss

### everything else is the same bewteen these two functions, the only difference
### is whether the training labels are self-generated using prediction or explicitly given

def self_d2_test(model, train, test_l, test_h, figType, lr_s = 1, lr_f = 10, batch_size = 40, epoch = 1):
    if (np.shape(train[0])[0]) % batch_size == 0:
        split_size = ceil((np.shape(train[0])[0])/batch_size)
        if split_size == 1: # batch size smaller than row
            raise "In this experiment, batch size shouldn't equal trainin sample size"
        else:
            pred_l = []
            pred_h = []
            acc = []
            loss = []
            history = EpochLogger(test_l, test_h)
            for e in range(epoch):
                np.random.shuffle(train[0])
                train_input, train_labels = paired_shuffle_split(train[0], train[1], split_size)
                batch_list = list(range(0, split_size))
                np.random.shuffle(batch_list)
                for row in batch_list:
                    pred_label = np.around(model.predict(train_input[row]), 0).astype(int)
                    # acc.append(np.sum(np.equal(pred_label, train_labels[row]))/len(pred_label))
                    # print(pred_label[-1])
                    # if row == 0:
                    #     pred_label[-1,:] = 1 - pred_label[-1,:]
                    model.fit(train_input[row], pred_label, callbacks = [history])
                    acc.append(history.acc[0])
                    loss.append(history.losses[0])
                    pred_l.append(history.pred_l[0])
                    pred_h.append(history.pred_h[0])
                    # pred_l.append(model.predict(test_l)[0][0])
                    # pred_h.append(model.predict(test_h)[0][0])
    else:
        raise "Batch size not divisible by sample size"

    return pred_l[-1], pred_h[-1], pred_l, pred_h, acc, loss


def mean_confidence_interval(data, confidence=0.95):
    ###assuming the input is an np array
    n = len(data)
    p = np.shape(data)[1]
    idx = np.array_split(list(range(p)), 3)
    low = data[:, idx[0]]
    high = data[:, idx[1]]
    loss = data[:, idx[2]]
    
    m_l, se_l = np.mean(low, axis = 0), scipy.stats.sem(low)
    m_h, se_h = np.mean(high, axis = 0), scipy.stats.sem(high)
    # m_acc, se_acc = np.mean(acc, axis = 0), scipy.stats.sem(acc)
    m_ls, se_ls = np.mean(loss, axis = 0), scipy.stats.sem(loss)
    #h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return (m_l, se_l, m_h, se_h, m_ls, se_ls)##change this to h if you want CIs.


def plot_mean_and_CI(mean, se, mean2 = None, se2 = None, mean3 = None, se3 = None,
                    color_shading='orange', color_shading2='purple', color_shading3 = 'red'):
    # plot the shaded range of the confidence intervals
    x = range(mean.shape[0])
    fig, (ax1, ax2) = plt.subplots(2)
    
    ax1.fill_between(range(mean.shape[0]), mean-se, mean+se,
                     color=color_shading, alpha=.5)
    ax1.fill_between(range(mean2.shape[0]), mean2-se2, mean2+se2,
                 color=color_shading2, alpha=.5)
    ax1.plot(x, mean, color = color_shading)
    ax1.plot(x, mean2, color = color_shading2)
    
    ax1.set_ylim((0,1))
    ax1.legend(['LowF0', 'HighF0'], loc = 'lower right') 
    
    ax2.fill_between(range(mean3.shape[0]), mean3-se3, mean3+se3,
                 color=color_shading3, alpha=.5)
    ax2.plot(x, mean3, color = color_shading3)
    ax2.set_ylim((0,0.03))
    ax2.legend(['Loss'], loc = 'lower right') 
    
    # plt.fill_between(range(mean3.shape[0]), mean3-se3, mean3+se3,
    #                  linestyle = '--', color=color_shading3, alpha=.5)
    
    # plt.fill_between(range(mean4.shape[0]), mean4-se4, mean4+se4,
    #              linestyle = '--', color=color_shading4, alpha=.5)
    

    # plot the mean on top
    # plt.plot(x, mean3)
    # plt.plot(x, mean4)
    
    return fig

# def plot_exp_results(fname, lrr = 12, c1 = 'blue', c2 = 'orange'):
#     r = []
#     with open(fname, 'rb') as f:
#         try:
#             while True:
#                 r.append(pkl.load(f))
#         except EOFError:
#             pass
#     r = np.array(r)
#     rev_l_m, rev_l_se, rev_h_m, rev_h_se = mean_confidence_interval(r)
#     f2 = plot_mean_and_CI(rev_l_m, rev_l_se, mean2 = rev_h_m, se2 = rev_h_se, color_shading = c1,
#                                     color_shading2=c2)
#     f2name = train_fig_dir+'%s_2_f_prob_%s_newdata.png'%(fname, lrr)
#     f2.savefig(f2name)


def plot_exp_results(fname, lrr = 12, c1 = 'orange', c2 = 'purple', c3 = 'red'):
    r = []
    filename = fname + '.pkl'
    with open(filename, 'rb') as f:
        try:
            while True:
                r.append(pkl.load(f))
        except EOFError:
            pass
    r = np.array(r)
    rev_l_m, rev_l_se, rev_h_m, rev_h_se, ls_m, ls_se = mean_confidence_interval(r)
    f2 = plot_mean_and_CI(rev_l_m, rev_l_se, rev_h_m, rev_h_se, ls_m, ls_se,
                          color_shading = c1, color_shading2=c2, color_shading3 = c3)
    f2name = train_fig_dir+'%s_Reg_2_f_%s_.png'%(fname, lrr)
    f2.savefig(f2name)


def compare_weights_2(model, train, batch_size, epoch_size, pred_l, pred_h, id, xlab, ylab):
    
    # get all weights before exposing
    before_weight_dict = {}
    for layer in model.layers:
        name = layer.get_config()['name']
        if layer.get_weights() != []:
            weights = layer.get_weights()[0]
            before_weight_dict[name] = weights
    
    result = {}
    result["slow"] = []
    result["s2"] = []
    result["fast"] = []
    result["f2"] = []
    result["output"] = []
    
    # almost no weight changes after 30th batch
    for i in range(0, np.shape(train[0])[0]):
        # expose model 1 batch
        model.fit(np.array([train[0][i]]), np.array([train[1][i]]),batch_size = 1, 
                  epochs = 1, verbose = 0)
    
        # get weights after exposing
        after_weight_dict = {}
        for layer in model.layers:
            name = layer.get_config()['name']
            if layer.get_weights() != []:
                weights = layer.get_weights()[0]
                after_weight_dict[name] = weights
    
        # add weights to result dictionary
        if (i % 1 == 0):
            for key in after_weight_dict:
                if key in result:
                    # trim weight array by "by" (we cant plot all 900 weights)
                    if ((key == "slow") or (key == "fast")):
                        by = 50
                    else: by = 1
                    result[key] = np.append(result[key], (after_weight_dict[key]).flatten()[::by])
                    
                #after = (after_weight_dict[key]).flatten()
                #before = (before_weight_dict[key]).flatten()
                #result[key] = result[key] + [np.sum(np.absolute(np.subtract(after,before)))]
    
    print("Exposure phase done")
    print("Start plot")
    print('%s TEST PROBABILITIES = '%(id), model.predict(pred_l)[0], model.predict(pred_h)[0])
    # plot 3d weight change plot
    # x = batch
    # y = weight index
    # z = weight value
    
    # slow_1
    
    # batchLen = len(train[0])//10
    # weightLen = int(len(result["slow"])/batchLen)

    batchLen = np.shape(train[0])[0]
    weightLen = int(len(result["slow"])/batchLen)
    
# scatter 3d plot
   
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    x = np.repeat(np.arange(1,batchLen+1),weightLen)
    y = np.tile(np.arange(1, weightLen+1), batchLen)*50
    z_slow = result["slow"]
    z_fast = result["fast"]
   
    print (len(x), len(y), len(z_slow))
    ax = fig.add_subplot(1,2,1,projection = '3d')
    ax.scatter3D(x,y,z_slow,c=z_slow,cmap='Blues')
    ax.set_title(id+" exposure : Slow layer weights")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Weight index")
    ax.set_zlabel("Weight value")
    
    ax = fig.add_subplot(1,2,2,projection = '3d')
    ax.scatter3D(x,y,z_fast,c=z_fast,cmap='Blues')
    ax.set_title(id+" exposure : Fast layer weights")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Weight index")
    ax.set_zlabel("Weight value")
    plt.show()
    
    return


