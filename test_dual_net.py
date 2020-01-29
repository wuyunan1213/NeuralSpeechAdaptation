########################################################################
#######       DO NOT MODIFY, DEFINITELY READ ALL OF THIS         #######
########################################################################

import numpy as np
import dual_net
import pickle
import copy
import random

def get_dualnet():
    """Define dualnet

    Explanation of parameters:
    type: layer type, supports data(input), IP(hidden units) and 
    channel: input channel
    num: output channel
    k: convolution kernel width (== height)
    group: split input channel into several groups, not used in this assignment
    """

    layers = {}
    layers[1] = {}
    layers[1]['type'] = 'DATA'
    layers[1]['height'] = 28
    layers[1]['width'] = 28
    layers[1]['batch_size'] = 64

    layers[2] = {}
    layers[2]['type'] = 'IP'
    layers[2]['num'] = 20 ###try first with 20 hidden units
    layers[2]['init_type'] = 'uniform'

    layers[3] = {}
    layers[3]['type'] = 'RELU'

    layers[4] = {}
    layers[4]['type'] = 'LOSS'
    layers[4]['num'] = 10
    return layers


def main():
    # define dualnet
    layers = get_dualnet()

    # load data
    # change the following value to true to load the entire dataset
    fullset = False
    xtrain, ytrain, xval, yval, xtest, ytest = dual_net.load_mnist(fullset)

    xtrain = np.hstack([xtrain, xval])
    ytrain = np.hstack([ytrain, yval])
    m_train = xtrain.shape[1]

    # NN parameters
    batch_size = 64
    mu = 0.9
    epsilon = 0.01
    gamma = 0.0001
    power = 0.75
    weight_decay = 0.0005
    w_lr = 1
    b_lr = 2

    test_interval = 200
    display_interval = 50
    snapshot = 500
    max_iter = 10000

    # initialize parameters
    params = dual_net.init_dualnet(layers)
    param_winc = copy.deepcopy(params)

    for l_idx in range(1, len(layers)):
        param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
        param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

    # learning iterations
    indices = range(m_train)
    random.shuffle(indices)
    for step in range(max_iter):
        # get mini-batch and setup the cnn with the mini-batch
        start_idx = step * batch_size % m_train
        end_idx = (step+1) * batch_size % m_train
        if start_idx > end_idx:
            random.shuffle(indices)
            continue
        idx = indices[start_idx: end_idx]

        [cp, param_grad] = dual_net.dual_net(params,
                        layers,
                        xtrain[:, idx],
                        ytrain[idx])

        # we have different epsilons for w and b
        w_rate = dual_net.get_lr(step, epsilon*w_lr, gamma, power)
        b_rate = dual_net.get_lr(step, epsilon*b_lr, gamma, power)
        params, param_winc = dual_net.sgd_momentum(w_rate,
                    b_rate,
                    mu,
                    weight_decay,
                    params,
                    param_winc,
                    param_grad)

        # display training loss
        if (step+1) % display_interval == 0:
            print 'cost = %f training_percent = %f' % (cp['cost'], cp['percent'])

        # display test accuracy
        if (step+1) % test_interval == 0:
            layers[1]['batch_size'] = xtest.shape[1]
            cptest, _ = dual_net.dual_net(params, layers, xtest, ytest)
            layers[1]['batch_size'] = 64
            print '\ntest accuracy: %f\n' % (cptest['percent'])

        # save params peridocally to recover from any crashes
        if (step+1) % snapshot == 0:
            pickle_path = 'dualnet.mat'
            pickle_file = open(pickle_path, 'wb')
            pickle.dump(params, pickle_file)
            pickle_file.close()


if __name__ == '__main__':
    main()
