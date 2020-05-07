######################################################################################
###    This model only has one slow/fast layer. Also, the pretrained model is exactly the same
###### as the model used in the exposure training for canonical/reverse
######################################################################################

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
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
# from tensorflow.keras.callbacks import Callback

from keras.callbacks import LambdaCallback, ModelCheckpoint

###import config and helper functions
import config
import generate
import train_2_full_rcr as train_one


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
PRETRAIN_EPOCHs = 10
EXPOSURE_BATCH_SIZE = 1
EXPOSURE_EPOCHS = 1

### Load all Data
# Labels are 1 = "b" and 0 = "p"
#print(data_dir)
pretrain_data = train_one.loadOfInt('pretrain.pkl', data_dir)

canonical = train_one.loadOfInt('canonical.pkl', data_dir)

rev = train_one.loadOfInt('rev.pkl', data_dir)

# rev2 = train_one.loadOfInt("rev2.pkl")
low_d2_test = np.array(train_one.loadOfInt('test.pkl', data_dir)[0])
high_d2_test = np.array(train_one.loadOfInt('test.pkl', data_dir)[2])

b_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[0])
p_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[2])

### In this version, I changed the activation to linear units, which is set as default in my implementation
### I also unfreeze the slow weights so that there's weight update in the slow pathway as well during exposure
lr_slow = 1
lr_fast = 12

slow_model = train_one.ff_nn_one(lr_s = 1, lr_f = 1)

history = train_one.Test_NBatchLogger(test_l = low_d2_test, test_h = high_d2_test, 
                                    test_b = b_d1_test, test_p = p_d1_test)
slow_hist = slow_model.fit(
    pretrain_data[0], pretrain_data[1],
    batch_size = PRETRAIN_BATCH_SIZE,
    epochs = PRETRAIN_EPOCHs,
    validation_data=(pretrain_data[2], pretrain_data[3]), 
    callbacks = [history]
    #callbacks=[tensorboard]
)

fig, (ax1, ax2, ax3) = plt.subplots(3)

print(np.shape(history.acc))
ax1.plot(history.acc)
ax1.plot(history.losses)
ax1.set_title('Training accuracy')
ax1.set(xlabel = 'Batch', ylabel = 'Accuracy')
ax1.legend(['Accuracy', 'Loss'], loc = 'lower left')

ax2.plot(history.pred_l)
ax2.plot(history.pred_h)
ax2.set_title('Testing_Probability_D2')
ax2.set_ylim((0,1))
ax2.set(xlabel = 'Batch', ylabel = 'Probability')
ax2.legend(['Low_F0', 'High_F0'], loc = 'lower left')

ax3.plot(history.pred_b)
ax3.plot(history.pred_p)
ax3.set_title('Testing_Probability_D1')
ax3.set_ylim((0,1))
ax3.set(xlabel = 'Batch', ylabel = 'Probability')
ax3.legend(['B_VOT', 'P_VOT'], loc = 'lower left')

plt.tight_layout()
figname = train_fig_dir + 'Training_Accuracy_2_layer.png'
plt.savefig(figname)
plt.close()

# fs_r2 = set_fs_weights(slow_model)
# fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, train_slow = True)
# train_one.compare_weights_2(fs, rev, 1, 1, low_d2_test, high_d2_test, "Reverse","before expose","after expose")

# new = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, train_slow = True)
# train_one.compare_weights_2(new, canonical, 1, 1, low_d2_test, high_d2_test, "Canonical","before expose","after expose")
# right now it seems like the slow weights are very stable, but in the canonical block, the fast weights are very unstable, 
# possibly leading to the unstableness in the canonical block.


n_exp = 1
for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, train_slow = True)
    r_l1, r_h1, rev_l1, rev_h1, r_b1, r_p1, rev_b1, rev_p1,  = train_one.test_d2_reliance(fs, rev, low_d2_test, high_d2_test, 
                                    b_d1_test, p_d1_test, 'REVERSE1', 
                                    lr_s = lr_slow, lr_f = lr_fast, 
                                    batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE1 = ', r_l1, r_h1, r_b1, r_p1)
    
    new = train_one.set_fs_weights_one(fs, lr_s = lr_slow, lr_f = lr_fast, train_slow = True)
    c_l, c_h, can_l, can_h, c_b, c_p, can_b, can_p = train_one.test_d2_reliance(new, canonical, low_d2_test, high_d2_test,
                                        b_d1_test, p_d1_test, 'CANONICAL1', 
    									lr_s = lr_slow, lr_f = lr_fast, 
    									batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('CANONICAL = ', c_l, c_h, c_b, c_p)
    new2 = train_one.set_fs_weights_one(new, lr_s = lr_slow, lr_f = lr_fast, train_slow = True)
    r_l2, r_h2, rev_l2, rev_h2, r_b2, r_p2, rev_b2, rev_p2  = train_one.test_d2_reliance(new2, rev, low_d2_test, high_d2_test, 
                                b_d1_test, p_d1_test, 'REVERSE2', 
                                lr_s = lr_slow, lr_f = lr_fast, 
                                batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE2 = ', r_l2, r_h2, r_b2, r_p2)


    
    t1 = can_l + can_h
    t2 = rev_l1 + rev_h1
    t3 = rev_l2 + rev_h2

    t4 = can_b + can_p 
    t5 = rev_b1 + rev_p1
    t6 = rev_b2 + rev_p2
    
    pkl.dump(t1, open('can_test.pkl', "ab"))
    pkl.dump(t2, open('rev_test1.pkl', "ab"))
    pkl.dump(t3, open('rev_test2.pkl', "ab"))

    pkl.dump(t4, open('can_bp.pkl', "ab"))
    pkl.dump(t5, open('rev_bp1.pkl', "ab"))
    pkl.dump(t6, open('rev_bp2.pkl', "ab"))



file_list = ['rev_test1', 'can_test', 'rev_test2', 'rev_bp1', 'can_bp', 'rev_bp2',]
for file in file_list:
    train_one.plot_exp_results(file, lrr = lr_fast)




