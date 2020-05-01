####This is a NN written from scratch

import numpy as np
import math
import scipy.io
import copy

def init_NN(layers):
	"""Initialize parameters of each layer in NN

	Args:
		layers: a dictionary that defines NN

	Returns:
		params: a dictionary stores initialized params
	"""

	params = {}

	#starts with second layer, first layer is data layer
	for i in range(2, len(layers)+1):
		params[i-1] = {}
		if layers[i]['type'] == 'IP':
			##Not uniform weights
			##check the number of units
			params[i-1]['w'] = scale*np.random.randn((N, layers[i]['num']))
			params[i-1]['b'] = scale*np.random.randn((N, ))

		elif layers[i]['type'] == 'LOSS':
			num = layers[i]['num']
			params[i]['w'] = np.random.rand(N, )
			params[i]['b'] = np.zeros(num - 1)

	return params

def inner_product_forward(input, layer, param):
    """Fully connected layer forward

    Args:
        input: a dictionary contains input data and shape information
        layer: one cnn layer, defined in testLeNet.py
        param: parameters, a dictionary

    Returns:
        output: a dictionary contains output data and shape information
    """
    num = layer['num']
    batch_size = input['batch_size']

    output = {}
    output['units'] = num
    output['batch_size'] = batch_size
    output['data'] = np.zeros((num, batch_size))

    input_n = {}
    for n in batch_size:
    	data = range(batch_size)
    	lin = data.dot(param['w']) + param['b']
    	output['data'][:, n] = lin

	assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'
	return output











