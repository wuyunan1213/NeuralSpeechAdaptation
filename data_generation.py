import pickle as pkl
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

###import config and helper functions for data generation
import config
import generate

###import paths and parameters
root_dir = config.root_dir
data_dir = config.data_dir
data_fig_dir = config.data_fig_dir

pre_file = config.pre_file
can_file = config.can_file
rev_file = config.rev_file
test_file = config.test_file
test_file_hor = config.test_file_hor
grid_file = config.grid_file

n_sub = config.n_sub
n_samples = config.n_samples
n_exposure = config.n_exposure

###Pre-training and Canonical params
mu_b = config.mu_b
mu_p = config.mu_p

sigma_b = config.sigma_b
sigma_p = config.sigma_p
sigma_b_exposure = config.sigma_b_exposure
sigma_p_exposure = config.sigma_p_exposure

sigma_sub = config.sigma_sub

###Reverse
mu_rev_b = config.mu_rev_b
mu_rev_p = config.mu_rev_p

###Test
mu_low = config.mu_low
mu_high = config.mu_high

sigma_low = config.sigma_low
sigma_high = config.sigma_high

###Input data generation
#pre-training
np.random.seed(10)
pr_data = generate.simData(n_sub, n_samples, mu_b, mu_p, sigma_b, sigma_p, sigma_sub, file = pre_file)
pkl.dump(pr_data, open(pre_file, "wb"))

#canonical
# np.random.seed(500)
# can_data = generate.simData(n_sub, n_exposure, mu_b, mu_p, sigma_b_exposure, sigma_p_exposure, sigma_sub, file = can_file)
import itertools
import generate
step = 9
d1 = np.linspace(0.1, 0.3, step)
d2 = np.linspace(0.1, 0.3, step)
d3 = np.linspace(0.7, 0.9, step)
d4 = np.linspace(0.7, 0.9, step)
l = np.asarray(list(itertools.product(d1,d2)))
l2 = np.asarray(list(itertools.product(d3,d4)))

grid_samples = map(lambda x: np.random.multivariate_normal(x, sigma_sub, size=[n_sub, ]), l)
grid_samples2 = map(lambda x: np.random.multivariate_normal(x, sigma_sub, size=[n_sub, ]), l2)
can = np.concatenate((grid_samples, grid_samples2))
can_data = generate.unpack_sample(can)
pkl.dump(can_data, open(can_file, "wb"))

#reverse
# np.random.seed(30000000)
# r_data = generate.simData(n_sub, n_exposure, mu_rev_b, mu_rev_p, sigma_b_exposure, sigma_p_exposure, sigma_sub, file = rev_file)
l = np.asarray(list(itertools.product(d1,d3)))
l2 = np.asarray(list(itertools.product(d2,d4)))

grid_samples = map(lambda x: np.random.multivariate_normal(x, sigma_sub, size=[n_sub, ]), l)
r_data = generate.unpack_sample(grid_samples)
pkl.dump(r_data, open(rev_file, "wb"))

#test
np.random.seed(1000)
test_data = generate.simTestData(n_sub,  sigma_sub, file = test_file)
pkl.dump(test_data, open(test_file, "wb"))

#test along the primary dimension
np.random.seed(1000)
test_data_hor = generate.simTestData_hor(n_sub,  sigma_sub, file = test_file_hor)
pkl.dump(test_data_hor, open(test_file_hor, "wb"))

np.random.seed(1000)
grid = generate.grid(sigma_sub, n_sub, step = 7)
pkl.dump(grid, open(grid_file, "wb"))

#full grid to look at model's weighting of acoustic dimensions





