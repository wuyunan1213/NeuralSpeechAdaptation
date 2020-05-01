import VisualizeNN as VisNN
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from numpy.random import random_sample

network_structure = [3,4,1]
# Draw the Neural Network without weights

# Draw the Neural Network with weights

vec = [-0.6, -0.7, -0.8, -0.9, -1. , -2]

weights = []

for i in range(5):
    weight1 = random_sample((3,4))
    weight2 = random_sample((4,1))
    w = [weight1, weight2]
    weights.append(w)

    #coefs[i][0][0][1]

network=VisNN.DrawNN(network_structure)
fig = network.draw()

network=VisNN.DrawNN(network_structure, weights[0])
network.draw()

network=VisNN.DrawNN(network_structure, weights)
network.animate(weights)

