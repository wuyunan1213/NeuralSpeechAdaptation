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
fig_dir = config.fig_dir

pre_file = config.pre_file
can_file = config.can_file
rev_file = config.rev_file
test_file = config.test_file

n_sub = config.n_sub
n_samples = config.n_samples
n_exposure = config.n_exposure

###Pre-training and Canonical params
mu_b = config.mu_b
mu_p = config.mu_p

sigma_b = config.sigma_b
sigma_p = config.sigma_p
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
np.random.seed(10000000)
can_data = generate.simData(n_sub, n_exposure, mu_b, mu_p, sigma_b, sigma_p, sigma_sub, file = can_file)
pkl.dump(can_data, open(can_file, "wb"))

#reverse
np.random.seed(30000000)
r_data = generate.simData(n_sub, n_exposure, mu_rev_b, mu_rev_p, sigma_b, sigma_p, sigma_sub, file = rev_file)
pkl.dump(r_data, open(rev_file, "wb"))

#test
np.random.seed(1000)
test_data = generate.simTestData(n_sub,  sigma_sub, file = test_file)
pkl.dump(test_data, open(test_file, "wb"))





