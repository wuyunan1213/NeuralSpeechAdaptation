""" This experiment explores self-supervision: after pre-training (the model
is trained as usual with explicit labels during pre-training),  the model generates labels for itself (using model.predict)
rather than trains on explicit labels during exposure. This creates a model that uses its learned representations to
train itself during the exposure phase, simulating human behavior. 
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

from keras.callbacks import LambdaCallback, ModelCheckpoint

###import config and helper functions
import config
import generate
import Self_Reg_train_2_full_rcr as train_one

###suppresses the runtime error. Not sure what happened but this works
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###import paths and parameters
root_dir = config.root_dir
data_dir = config.data_dir
train_fig_dir = config.train_fig_dir
data_fig_dir = config.data_fig_dir

###Training parameters
PRETRAIN_BATCH_SIZE = 10
PRETRAIN_EPOCHs = 10
EXPOSURE_BATCH_SIZE = 10
EXPOSURE_EPOCHS = 5

penalty = 0.0009

### Load all Data
# Labels are 0 = "b" and 1 = "p"
#print(data_dir)
pretrain_data = train_one.loadOfInt('pretrain.pkl', data_dir)

canonical = train_one.loadOfInt('canonical.pkl', data_dir)

rev = train_one.loadOfInt('rev.pkl', data_dir)

# rev2 = train_one.loadOfInt("rev2.pkl")
low_d2_test = np.array(train_one.loadOfInt('test.pkl', data_dir)[0])
high_d2_test = np.array(train_one.loadOfInt('test.pkl', data_dir)[2])

b_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[0])
p_d1_test = np.array(train_one.loadOfInt('test_hor.pkl', data_dir)[2])

### In this version, default activation is linear
### I also unfreeze the slow weights so that there's weight update in the slow pathway as well during exposure

slow_model = train_one.ff_nn_one(lr_s = 1, lr_f = 1, penalty = penalty)

history = train_one.Test_NBatchLogger(test_l = low_d2_test, test_h = high_d2_test)
slow_hist = slow_model.fit(
    pretrain_data[0], pretrain_data[1],
    batch_size = PRETRAIN_BATCH_SIZE,
    epochs = PRETRAIN_EPOCHs,
    validation_data=(pretrain_data[2], pretrain_data[3]), 
    callbacks = [history]
)

# outputSLOW = fs.get_layer('output').get_weights()[0][10:20]
# outputFAST = fs.get_layer('output').get_weights()[0][60:70]
# SLOW = fs.get_layer('slow').get_weights()[0][0,30:35]
# FAST = fs.get_layer('fast').get_weights()[0][0,30:35]
lr_slow = 1
lr_fast = 90
j = lr_fast

n_exp = 20
for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l1, r_h1, rev_l1, rev_h1  = train_one.test_d2_reliance(fs, rev, low_d2_test, high_d2_test, 'REVERSE1', 
                                    lr_s = lr_slow, lr_f = lr_fast, 
                                    batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE1 = ', r_l1, r_h1)
    
    new = train_one.set_fs_weights_one(fs, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    c_l, c_h, can_l, can_h = train_one.test_d2_reliance(new, canonical, low_d2_test, high_d2_test, 'CANONICAL1', 
    									lr_s = lr_slow, lr_f = lr_fast, 
    									batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('CANONICAL = ', c_l, c_h)
    new2 = train_one.set_fs_weights_one(new, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l2, r_h2, rev_l2, rev_h2  = train_one.test_d2_reliance(new2, rev, low_d2_test, high_d2_test, 'REVERSE2', 
                                lr_s = lr_slow, lr_f = lr_fast, 
                                batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE2 = ', r_l2, r_h2)


    
    t1 = can_l + can_h
    t2 = rev_l1 + rev_h1
    t3 = rev_l2 + rev_h2
    
    pkl.dump(t1, open('%scan_test_self.pkl'%(j), "ab"))
    pkl.dump(t2, open('%srev_test1_self.pkl'%(j), "ab"))
    pkl.dump(t3, open('%srev_test2_self.pkl'%(j), "ab"))



file_list = ['%srev_test1_self'%(j), '%scan_test_self'%(j), '%srev_test2_self'%(j)]
for file in file_list:
    train_one.plot_exp_results(file, lrr = lr_fast)




