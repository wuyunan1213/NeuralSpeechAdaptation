""" This experiment compares the self-superved model and the supervised model: after pre-training (the model
is trained as usual with explicit labels during pre-training), the exposure data are either trained
using self-generated labels (using model.predict) or given labels (defined in the simulated data)
Note that the training procedure was made identical for both models except for the output label. Thus there were some 
changes for the supervised model that were not in previous versions.
"""

import pickle as pkl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras_lr_multiplier import LRMultiplier
from keras.callbacks import Callback
import os
import random
from math import ceil
from keras.callbacks import LambdaCallback, ModelCheckpoint

###import config and helper functions
import config
import generate
import compare_train1 as train_one
from scipy import stats
import pandas as pd
from keras.models import load_model



###suppresses the runtime error. Not sure what happened but this works
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TF_KERAS']='True'
###import paths and parameters
root_dir = config.root_dir
data_dir = config.data_dir
train_fig_dir = config.train_fig_dir
data_fig_dir = config.data_fig_dir

###Training parameters
PRETRAIN_BATCH_SIZE = 10
PRETRAIN_EPOCHs = 7
EXPOSURE_BATCH_SIZE = 10
EXPOSURE_EPOCHS = 10

penalty = 0.0004
lr_slow = 1
lr_fast = 91

pretrain_data = train_one.loadOfInt('pretrain.pkl', data_dir)
# pretrain_data = train_one.loadOfInt('pretrain1.pkl', data_dir)
# pretrain_data = train_one.loadOfInt('pretrain2.pkl', data_dir)

canonical = train_one.loadOfInt('canonical.pkl', data_dir)

rev = train_one.loadOfInt('rev.pkl', data_dir)

# rev2 = train_one.loadOfInt("rev2.pkl")
# use dimension1 = .49 instead of .5 achieves prettier results
low_d2_test = np.array(train_one.loadOfInt('test5.pkl', data_dir)[0])
high_d2_test = np.array(train_one.loadOfInt('test5.pkl', data_dir)[2])

# b_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[0])
# p_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[2])

### In this version, I changed the activation to linear units, which is set as default in my implementation
### I also unfreeze the slow weights so that there's weight update in the slow pathway as well during exposure

slow_model = train_one.ff_nn_one(lr_s = 1, lr_f = lr_fast, penalty = penalty)
# slow_model.load_weights('sm_weights')
history = train_one.Test_NBatchLogger(test_l = low_d2_test, test_h = high_d2_test)
slow_hist = slow_model.fit(
    pretrain_data[0], pretrain_data[1],
    batch_size = PRETRAIN_BATCH_SIZE,
    epochs = PRETRAIN_EPOCHs,
    validation_data=(pretrain_data[2], pretrain_data[3]), 
    callbacks = [history]
    #callbacks=[tensorboard]
)


# slow_model.save('slow_model')

# fig, (ax1, ax2) = plt.subplots(2)

# print(np.shape(history.acc))
# ax1.plot(history.acc)
# ax1.plot(history.losses)
# ax1.set_title('Training accuracy')
# ax1.set(xlabel = 'Epoch', ylabel = 'Accuracy')
# ax1.legend(['Accuracy', 'Loss'], loc = 'lower left')

# ax2.plot(history.pred_l)
# ax2.plot(history.pred_h)
# ax2.set_title('Testing_Probability')
# ax2.set_ylim((0,1))
# ax2.set(xlabel = 'Epoch', ylabel = 'Probability')
# ax2.legend(['Low_F0', 'High_F0'], loc = 'lower left')


# plt.tight_layout()
# figname = train_fig_dir + 'Training_Accuracy_2_layer.png'
# plt.savefig(figname)
# plt.close()

# outputSLOW = fs.get_layer('output').get_weights()[0][10:20]
# outputFAST = fs.get_layer('output').get_weights()[0][60:70]
# SLOW = fs.get_layer('slow').get_weights()[0][0,30:35]
# FAST = fs.get_layer('fast').get_weights()[0][0,30:35]
n_exp = 1

'''
SUPERVISED MODEL
'''
j = 4
# slow_model.save_weights('sm_weights1')

'''
SELF-SUPERVISED MODEL
'''

for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    self_fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l1, r_h1, rev_l1, rev_h1, rev1_acc, rev1_ls  = train_one.self_d2_test(self_fs, rev, low_d2_test, high_d2_test, 'Self_REVERSE1', 
                                    lr_s = lr_slow, lr_f = lr_fast, 
                                    batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE1 = ', r_l1, r_h1)


    t2 = rev_l1 + rev_h1 + rev1_ls
    acc = np.mean(rev1_acc)
    pkl.dump(t2, open('%s_%s_Acc_%s.pkl'%(j, lr_fast, acc), "ab"))


file_list = ['%s_%s_Acc_%s'%(j, lr_fast, acc)]
for file in file_list:
    train_one.plot_exp_results(file, lrr = lr_fast)

plt.plot(rev_l1)
plt.plot(rev_h1)


l = [r_l1, c_l, r_l2]
h = [r_h1, c_h, r_h2]
fig, ax1 = plt.subplots()

ax1.plot(l)
ax1.plot(h)
ax1.set_title('Self-Supervised model')
ax1.set_ylim((0,1))
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(['Reverse1', 'Canonical', 'Reverse2'])
ax1.set(xlabel = 'Block', ylabel = 'Probability')
ax1.legend(['LowF0', 'HighF0'], loc = 'lower left')
figname = train_fig_dir + 'self_sf_test_dw_1.png'
fig.savefig(figname)




