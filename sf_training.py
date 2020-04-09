import pickle as pkl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

import os
import random
import keras
import tensorflow as tf
from keras_lr_multiplier import LRMultiplier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import Callback
from keras.callbacks import LambdaCallback, ModelCheckpoint

###import config and helper functions
import config
import generate
import train

###suppresses the runtime error. Not sure what happened but this works
os.environ['KMP_DUPLICATE_LIB_OK']='True'

###import paths and parameters
root_dir = config.root_dir
data_dir = config.data_dir
fig_dir = config.fig_dir

pre_file = config.pre_file
can_file = config.can_file
rev_file = config.rev_file
test_file = config.test_file

###Training parameters
batch_size = config.batch_size
epochs = config.epochs

### Load all Data
# Labels are 1 = "b" and 0 = "p"
#print(data_dir)
pretrain_data = train.loadOfInt('pretrain.pkl', data_dir)

canonical = train.loadOfInt('canonical.pkl', data_dir)
rev = train.loadOfInt('rev.pkl', data_dir)

# rev2 = train.loadOfInt("rev2.pkl")
low_d2_test = np.array(train.loadOfInt('test.pkl', data_dir)[0])
high_d2_test = np.array(train.loadOfInt('test.pkl', data_dir)[2])


###pre-training the model
slow_model = train.ff_nn()
history = train.NBatchLogger()
slow_hist = slow_model.fit(
    pretrain_data[0], pretrain_data[1],
    batch_size = batch_size,
    epochs = epochs,
    validation_data=(pretrain_data[2], pretrain_data[3]), 
    callbacks = [history]
    #callbacks=[tensorboard]
)
# plt.plot(history.acc)
# plt.plot(history.losses)
# plt.title('Training accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Batch')
# plt.legend(['Accuracy', 'Loss'], loc = 'upper left')
# figname = fig_dir + 'Training_Accuracy.png'
# plt.savefig(figname)
# plt.close()

EXPOSURE_BATCH_SIZE = 1
EXPOSURE_EPOCHS = 1

# ### Three different fs_nn with similar initializations
fs = train.set_fs_weights(slow_model)
# fs_r2 = set_fs_weights(slow_model)

# ### Exposure phase training with canonical and reverse data
c_l, c_h = train.test_d2_reliance(fs, canonical, low_d2_test, high_d2_test, 'CANONICAL', batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
print('CANONICAL = ', c_l, c_h)
r_l, r_h = train.test_d2_reliance(fs, rev, low_d2_test, high_d2_test, 'REVERSE', batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
print('REVERSE = ', r_l, r_h)
# r2_l, r2_h = test_d2_reliance(fs_r2, rev2, low_d2_test, high_d2_test, batch_size = EXPOSURE_BATCH_SIZE, epoch= EXPOSURE_EPOCHS)
# print(r1_l, r1_h)
# print(c_l, c_h)
# print(r2_l, r2_h)