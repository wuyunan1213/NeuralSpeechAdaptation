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
	N = layers[1]['num']
	#starts with second layer, first layer is data layer
	for i in range(2, len(layers)+1):
		params[i-1] = {}
		if layers[i]['type'] == 'IP_1':
			##Gaussian weights
			##first hidden layer after the input units
			##check the number of units
			params[i-1]['ws'] = np.random.randn((N, layers[i]['Snum']))
			params[i-1]['bs'] = np.random.randn((layers[i]['Snum'], ))

			params[i-1]['wf'] = np.random.randn((N, layers[i]['Fnum']))
			params[i-1]['bf'] = np.random.randn((layers[i]['Fnum'], ))

		if layers[i]['type'] == 'IP_n':
			##Gaussian weights
			##th nth hidden layer after the first hidden layer
			##check the number of units
			params[i-1]['ws'] = np.random.randn((params[i-1]['Snum'], layers[i]['Snum']))
			params[i-1]['bs'] = np.random.randn((params[i]['Snum'], ))

			params[i-1]['wf'] = np.random.randn((params[i-1]['Fnum'], layers[i]['Fnum']))
			params[i-1]['bf'] = np.random.randn((params[i]['Fnum'], ))

		elif layers[i]['type'] == 'LOSS':
			##hidden layer to loss layer
			##assuming there's at least one hidden layer branched 
			##into slow and fast and then connected to output
			num = layers[i]['num']
			params[i-1]['ws'] = np.random.randn((params[i-1]['Snum'], num))
			params[i-1]['bs'] = np.random.randn((num, ))

			params[i-1]['wf'] = np.random.randn((params[i-1]['Fnum'], num))
			params[i-1]['bf'] = np.random.randn((num, ))

	return params

def inner_product_forward(input, layer, param):
    """Fully connected layer forward

    Args:
        input: a dictionary contains input data and shape information
        layer: one NN layer, defined in test_NN.py
        param: parameters, a dictionary

    Returns:
        output: a dictionary contains output data and shape information
    """
    Snum = layer['Snum']
    Fnum = layer['Fnum']
    batch_size = input['batch_size']

    output = {}
    output['Snum'] = Snum
    output['Fnum'] = Fnum
    output['batch_size'] = batch_size
    output['Sdata'] = np.zeros((Snum, batch_size))
    output['Fdata'] = np.zeros((Fnum, batch_size))

    for n in range(batch_size):
    	data = input['data'][:,n]
    	Slin = data.dot(param['ws']) + param['bs']
    	Flin = data.dot(param['wf']) + param['bf']
    	output['Sdata'][:, n] = Slin
    	output['Fdata'][:, n] = Flin

	#assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'
	return output

def output_layer(input, layer, param):
    """Fully connected layer forward

    Args:
        input: a dictionary contains input data and shape information
        layer: one NN layer, defined in test_NN.py
        param: parameters, a dictionary

    Returns:
        output: a dictionary contains output data and shape information
    """
    Snum = input['Snum']
    Fnum = input['Fnum']
    batch_size = input['batch_size']

	Sdata = input['Sdata'][:,n]
	Fdata = input['Fdata'][:,n]

    N = layer['num']

    sigmoid = np.zeros((batch_size,))

    for n in range(batch_size):
    	Slin = -np.sum(Sdata.dot(param['ws']) + param['bs'])
    	Flin = -np.sum(Fdata.dot(param['wf']) + param['bf'])
    	sigmoid[n] = 1/(1+np.exp(Slin + Flin))

	#assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'
	return sigmoid

# def inner_product_forward_middle(input, layer, param):
#     """Fully connected layer forward

#     Args:
#         input: a dictionary contains input data and shape information
#         layer: one NN layer, defined in test_NN.py
#         param: parameters, a dictionary

#     Returns:
#         output: a dictionary contains output data and shape information
#     """
#     Snum = layer['Snum']
#     Fnum = layer['Fnum']
#     batch_size = input['batch_size']

#     output = {}
#     output['Snum'] = Snum
#     output['Fnum'] = Fnum
#     output['batch_size'] = batch_size
#     output['data'] = np.zeros((Snum+Fnum, batch_size))

#     for n in range(batch_size):
#     	data = input['data'][:,n]
#     	lin = data.dot(param['w']) + param['b']
#     	output['data'][:, n] = lin

# 	assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'
# 	return output


def inner_product_backward(output, input, layer, param):
    """Fully connected layer backward

    Args:
        output: a dictionary contains output data and shape information
        input: a dictionary contains input data and shape information
        layer: one cnn layer, defined in testLeNet.py
        param: parameters, a dictionary

    Returns:
        para_grad: a dictionary stores gradients of parameters
        input_od: gradients w.r.t input data
    """
    param_grad = {}
    param_grad['b'] = np.zeros(param['b'].shape)
    param_grad['w'] = np.zeros(param['w'].shape)
    input_od = np.zeros(input['data'].shape)

    # TODO: implement your inner product backward pass here
    # implementation begins
    data = input['data']
    param_grad['b'] += np.sum(output['diff'], axis = 1)
    param_grad['w'] += data.dot(output['diff'].T)
    input_od = param['w'].dot(output['diff'])
# implementation ends

    assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

    return param_grad, input_od


def bce_loss(sigmoid, output, y):
    """Loss layer

    Args:
        wb: concatenation of w and b, shape = (num+1, K-1)
        input: input data, shape = (num, batch size)
        y: ground truth label, shape = (batch size, )

    Returns:
        bce: binary cross entropy loss, a scalar, no need to divide batch size
        g: gradients of parameters
        od: gradients w.r.t input data
        percent: accuracy for this batch
    """		
	batch_size = input['batch_size']
	for n in range(batch_size):
		Sdata = -np.sum(input['Sdata'][:,n])
		Fdata = -np.sum(input['Fdata'][:,n])
		sigmoid = 1/(1+np.exp(Sdata + Fdata))
		label = y[n]
		loss = label*np.log(sigmoid) + (1-label)*np.log(1-sigmoid)

	# compute gradients
	od = sigmoid - 










