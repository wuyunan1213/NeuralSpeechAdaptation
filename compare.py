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
import compare_train as train_one
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
EXPOSURE_EPOCHS = 70

penalty = 0.0004
lr_slow = 1
lr_fast = 91

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

slow_model = train_one.ff_nn_one(lr_s = 1, lr_f = lr_fast, penalty = penalty)

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
# slow_model.save_weights('sm_weights')
# slow_model.load_weights('sm_weights')

# slow_model = load_model('slow_model')
for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l1, r_h1, rev_l1, rev_h1  = train_one.test_d2_reliance(fs, rev, low_d2_test, high_d2_test, 'Supervised_REVERSE1', 
                                    lr_s = lr_slow, lr_f = lr_fast, 
                                    batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE1 = ', r_l1, r_h1)
    
    new = train_one.set_fs_weights_one(fs, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    c_l, c_h, can_l, can_h = train_one.test_d2_reliance(new, canonical, low_d2_test, high_d2_test, 'Supervised_CANONICAL1', 
    									lr_s = lr_slow, lr_f = lr_fast, 
    									batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('CANONICAL = ', c_l, c_h)
    new2 = train_one.set_fs_weights_one(new, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l2, r_h2, rev_l2, rev_h2  = train_one.test_d2_reliance(new2, rev, low_d2_test, high_d2_test, 'Supervised_REVERSE2', 
                                lr_s = lr_slow, lr_f = lr_fast, 
                                batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE2 = ', r_l2, r_h2)


    
    t1 = can_l + can_h
    t2 = rev_l1 + rev_h1
    t3 = rev_l2 + rev_h2
    
    pkl.dump(t1, open('%scan_test_%s.pkl'%(j, lr_fast), "ab"))
    pkl.dump(t2, open('%srev_test1_%s.pkl'%(j, lr_fast), "ab"))
    pkl.dump(t3, open('%srev_test2_%s.pkl'%(j, lr_fast), "ab"))



file_list = ['%srev_test1_%s'%(j, lr_fast), '%scan_test_%s'%(j, lr_fast), '%srev_test2_%s'%(j, lr_fast)]
for file in file_list:
    train_one.plot_exp_results(file, lrr = lr_fast)

plt.plot(rev_l1)
plt.plot(rev_h1)
plt.ylim(((0,1)))
figname = train_fig_dir + 'sf_rev.png'
plt.savefig(figname)

# l = [r_l1, c_l, r_l2]
# h = [r_h1, c_h, r_h2]
# fig, ax1 = plt.subplots()

# ax1.plot(l)
# ax1.plot(h)
# ax1.set_title('Supervised model')
# ax1.set_ylim((0,1))
# ax1.set_xticks([0,1,2])
# ax1.set_xticklabels(['Reverse1', 'Canonical', 'Reverse2'])
# ax1.set(xlabel = 'Block', ylabel = 'Probability')
# ax1.legend(['LowF0', 'HighF0'], loc = 'lower left')
# figname = train_fig_dir + 'sf_test_dw.png'
# fig.savefig(figname)

'''

SELF-SUPERVISED MODEL

'''

for i in range(n_exp):
    # ### Exposure phase training with canonical and reverse data
    self_fs = train_one.set_fs_weights_one(slow_model, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l1, r_h1, rev_l1, rev_h1  = train_one.self_d2_test(self_fs, rev, low_d2_test, high_d2_test, 'Self_REVERSE1', 
                                    lr_s = lr_slow, lr_f = lr_fast, 
                                    batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE1 = ', r_l1, r_h1)
    
    self_new = train_one.set_fs_weights_one(self_fs, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    c_l, c_h, can_l, can_h = train_one.self_d2_test(self_new, canonical, low_d2_test, high_d2_test, 'Self_CANONICAL1', 
                                        lr_s = lr_slow, lr_f = lr_fast, 
                                        batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('CANONICAL = ', c_l, c_h)
    self_new2 = train_one.set_fs_weights_one(self_new, lr_s = lr_slow, lr_f = lr_fast, penalty = penalty, train_slow = True)
    r_l2, r_h2, rev_l2, rev_h2  = train_one.self_d2_test(self_new2, rev, low_d2_test, high_d2_test, 'Self_REVERSE2', 
                                lr_s = lr_slow, lr_f = lr_fast, 
                                batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
    print('REVERSE2 = ', r_l2, r_h2)


    t1 = can_l + can_h
    t2 = rev_l1 + rev_h1
    t3 = rev_l2 + rev_h2
    
    pkl.dump(t1, open('%scan_test_self_%s.pkl'%(j, lr_fast), "ab"))
    pkl.dump(t2, open('%srev_test1_self_%s.pkl'%(j, lr_fast), "ab"))
    pkl.dump(t3, open('%srev_test2_self_%s.pkl'%(j, lr_fast), "ab"))



file_list = ['%srev_test1_self_%s'%(j, lr_fast), '%scan_test_self_%s'%(j, lr_fast), '%srev_test2_self_%s'%(j, lr_fast)]
for file in file_list:
    train_one.plot_exp_results(file, lrr = lr_fast)

# plt.plot(rev_l1)
# plt.plot(rev_h1)


# l = [r_l1, c_l, r_l2]
# h = [r_h1, c_h, r_h2]
# fig, ax1 = plt.subplots()

# ax1.plot(l)
# ax1.plot(h)
# ax1.set_title('Self-Supervised model')
# ax1.set_ylim((0,1))
# ax1.set_xticks([0,1,2])
# ax1.set_xticklabels(['Reverse1', 'Canonical', 'Reverse2'])
# ax1.set(xlabel = 'Block', ylabel = 'Probability')
# ax1.legend(['LowF0', 'HighF0'], loc = 'lower left')
# figname = train_fig_dir + 'self_sf_test_dw.png'
# fig.savefig(figname)

# FAST = slow_model.get_layer('fast').get_weights()[0]
# SLOW = slow_model.get_layer('slow').get_weights()[0]
# f = np.absolute(FAST).flatten()
# s = np.absolute(SLOW).flatten()
# df = pd.DataFrame(columns = ['pathway', 'weight'])
# df['pathway'] = np.concatenate((np.repeat('fast', len(f)), np.repeat('slow', len(s))))
# df['weight'] = np.concatenate((f,s))
# df.to_csv(path_or_buf = root_dir + 'SLOW_FAST.csv', index = False)


# d1_w_slow = np.absolute(SLOW[:15,:])
# d2_w_slow = np.absolute(SLOW[15:30,:])
# d1_w_fast = np.absolute(FAST[:15,:])
# d2_w_fast = np.absolute(FAST[15:30,:])

# df = pd.DataFrame(columns = ['pathway', 'dimension', 'weight'])
# w1_slow = d1_w_slow.flatten()
# w2_slow = d2_w_slow.flatten()
# w1_fast = d1_w_fast.flatten()
# w2_fast = d2_w_fast.flatten()
# df['pathway'] = np.concatenate((np.repeat('fast', 2*len(w1_slow)), np.repeat('slow', 2*len(w1_slow))))
# df['dimension'] = np.concatenate((np.repeat('d1', len(w1_slow)), np.repeat('d2', len(w2_slow)),
#                                   np.repeat('d1', len(w1_slow)), np.repeat('d2', len(w2_slow))))
# df['weight'] = np.concatenate((w1_fast, w2_fast, w1_slow, w2_slow))
# df.to_csv(path_or_buf = root_dir + 'PATHWAY_DIM_pretrained.csv', index = False)



# df = pd.DataFrame(columns = ['dimension', 'weight'])
# w1_fast = d1_w_fast.flatten()
# w2_fast = d2_w_fast.flatten()
# df['dimension'] = np.concatenate((np.repeat('d1', len(w1_fast)), np.repeat('d2', len(w2_fast))))
# df['weight'] = np.concatenate((w1_fast, w2_fast))
# df.to_csv(path_or_buf = root_dir + 'FAST_pretrained.csv', index = False)


# s_FAST = new2.get_layer('fast').get_weights()[0]
# s_SLOW = new2.get_layer('slow').get_weights()[0]
# self_FAST = self_new2.get_layer('fast').get_weights()[0]
# self_SLOW = self_new2.get_layer('slow').get_weights()[0]

# s_d1 = np.absolute(s_FAST[:15,:])
# s_d2 = np.absolute(s_FAST[15:30,:])
# self_d1 = np.absolute(self_FAST[:15,:])
# self_d2 = np.absolute(self_FAST[15:30,:])

# df = pd.DataFrame(columns = ['model', 'dimension', 'weight'])
# sw1 = s_d1.flatten()
# sw2 = s_d2.flatten()
# selfw1 = self_d1.flatten()
# selfw2 = self_d2.flatten()

# df['model'] = np.concatenate((np.repeat('supervised', 2*len(w1_fast)), np.repeat('self_supervised', 2*len(w2_fast))))
# df['dimension'] = np.concatenate((np.repeat('d1', len(w1_fast)), np.repeat('d2', len(w2_fast)),
#                                   np.repeat('d1', len(w1_fast)), np.repeat('d2', len(w2_fast))))
# df['weight'] = np.concatenate((sw1, sw2, selfw1, selfw2))
# df.to_csv(path_or_buf = root_dir + 'MODEL_DIM.csv', index = False)




