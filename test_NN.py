import numpy as np
import cnn_lenet
import pickle
import copy
import random

import config
import generate
import train


root_dir = config.root_dir
data_dir = config.data_dir
train_fig_dir = config.train_fig_dir
data_fig_dir = config.data_fig_dir

pre_file = config.pre_file
can_file = config.can_file
rev_file = config.rev_file
test_file = config.test_file

def get_NN(batch_size = 1, n_input = 30, n_slow = 30, n_fast = 30, n_output = 1):
    """Define NN
    """

    layers = {}
    layers[1] = {}
    layers[1]['type'] = 'DATA'
    layers[1]['num'] = n_input
    layers[1]['batch_size'] = batch_size

    layers[2]['type'] = 'IP'
    layers[2]['Snum'] = n_slow
	layers[2]['Fnum'] = n_fast

	layers[3]['type'] = 'LOSS'
	layers[3]['num'] = n_output

	return layers

def main():
	layers = get_NN()

	#load data
	pretrain_data = train.loadOfInt('pretrain.pkl', data_dir)

	canonical = train.loadOfInt('canonical.pkl', data_dir)
	rev = train.loadOfInt('rev.pkl', data_dir)
