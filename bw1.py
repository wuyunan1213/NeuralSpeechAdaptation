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
import bw_train as train_one
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

penalty = 0.0009
lr_slow = 1
lr_fast = 91

# pretrain_data = train_one.loadOfInt('pretrain.pkl', data_dir)
# pretrain_data = train_one.loadOfInt('pretrain1.pkl', data_dir)
pretrain_data = train_one.loadOfInt('pretrain.pkl', data_dir)

canonical = train_one.loadOfInt('canonical.pkl', data_dir)

rev = train_one.loadOfInt('rev.pkl', data_dir)

# rev2 = train_one.loadOfInt("rev2.pkl")
# use dimension1 = .49 instead of .5 achieves prettier results
low_d2_test = np.array(train_one.loadOfInt('test5.pkl', data_dir)[0])
high_d2_test = np.array(train_one.loadOfInt('test5.pkl', data_dir)[2])

b_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[0])
p_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[2])

### In this version, I changed the activation to linear units, which is set as default in my implementation
### I also unfreeze the slow weights so that there's weight update in the slow pathway as well during exposure

slow_model = train_one.ff_nn_one(lr_s = 1, lr_f = lr_fast, penalty = penalty)
# history = train_one.Test_NBatchLogger(test_l = low_d2_test, test_h = high_d2_test)
# slow_hist = slow_model.fit(
#     pretrain_data[0], pretrain_data[1],
#     batch_size = PRETRAIN_BATCH_SIZE,
#     epochs = PRETRAIN_EPOCHs,
#     validation_data=(pretrain_data[2], pretrain_data[3]), 
#     callbacks = [history]
#     #callbacks=[tensorboard]
# )

# slow_model.save_weights('sm_weights')
slow_model.load_weights('sm_weights')



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
n_exp = 1

'''
SELF-SUPERVISED MODEL
'''

for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    self_fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    rev_l1, rev_h1, rev_short1, rev_long1, rev1_acc, rev1_ls  = train_one.self_d2_test(self_fs, rev, low_d2_test, high_d2_test, 'Self_REVERSE1', p_d1_test, b_d1_test, lr_s = lr_slow, lr_f = lr_fast, batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    
    print('REVERSE1 = ', rev_l1[-1], rev_h1[-1], rev_short1[-1], rev_long1[-1])


    self_new = train_one.set_fs_weights_one(self_fs, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    can_l, can_h, can_short, can_long, can_acc, can_ls = train_one.self_d2_test(self_new, canonical, low_d2_test, high_d2_test, 'Self_CANONICAL1', p_d1_test, b_d1_test, lr_s = lr_slow, lr_f = lr_fast, batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    
    print('CANONICAL = ', can_l[-1], can_h[-1], can_short[-1], can_long[-1])
    
    
    self_new2 = train_one.set_fs_weights_one(self_new, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    rev_l2, rev_h2, rev_short2, rev_long2,  rev2_acc, rev2_ls  = train_one.self_d2_test(self_new2, rev, low_d2_test, high_d2_test, 'Self_REVERSE2', p_d1_test, b_d1_test, lr_s = lr_slow, lr_f = lr_fast, batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    
    print('REVERSE2 = ', rev_l2[-1], rev_h2[-1], rev_short2[-1], rev_long2[-1])

l = [rev_l1[-1], can_l[-1], rev_l2[-1]]
h = [rev_h1[-1], can_h[-1], rev_h2[-1]]
long = [rev_long1[-1], can_long[-1], rev_long2[-1]]
short = [rev_short1[-1], can_short[-1], rev_short2[-1]]

fig, ax1 = plt.subplots()

ax1.plot(long, color = 'green', linestyle = '--')
ax1.plot(short, color = 'blue', linestyle = '--')

ax1.plot(h, color = 'purple', linestyle = '-')
ax1.plot(l, color = 'orange', linestyle = '-')



ax1.set_title('Self-Supervised model')
ax1.set_ylim((-0.2,1.2))
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(['Reverse1', 'Canonical', 'Reverse2'])
ax1.set(xlabel = 'Block', ylabel = 'Probability')
ax1.legend(['LongVOT', 'ShortVOT', 'HighF0', 'LowF0'], loc = 'lower left')

figname = train_fig_dir + 'self_bw1.png'
fig.savefig(figname)

plt.plot(rev_h2)
plt.plot(rev_l2)




