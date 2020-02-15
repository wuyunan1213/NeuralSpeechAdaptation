import numpy as np
import pickle as pkl
import datetime as dt
import matplotlib.pyplot as plt

### Sampling Sizes
n_osc = 15
n_samples = 10000000
files = ["b_samples_10m", "p_samples_10m"]

### Sampling Distribution Metrics
mu_b = np.array([0.25, 0.2])
mu_p = np.array([0.75, 0.8])
sigma_b = np.array([[0.006, 0], [0, 0.016]])
sigma_p = np.array([[0.006, 0], [0, 0.016]])
sigma_osc = np.array([[0.04, 0], [0, 0.04]])

'''
### Generate Plot
def data_dist(mu_b, sig_b, mu_p, sig_p, n, file = None):
    b_plot = np.random.multivariate_normal(mu_b, sig_b, size=[n,])
    p_plot = np.random.multivariate_normal(mu_p, sig_p, size=[n,])
    plt.scatter(b_plot[:,0],b_plot[:,1])
    plt.scatter(p_plot[:,0],p_plot[:,1])
    if type(file) == str:
        plt.savefig(file)
    else:
        plt.show()
data_dist(mu_b, sigma_b, mu_p, sigma_p, 5000, "data_dist.png")
'''

### Distribution Samples
b_centers = np.random.multivariate_normal(mu_b, sigma_b, size=[n_samples,])
p_centers = np.random.multivariate_normal(mu_p, sigma_p, size=[n_samples,])

### Quick'n'Dirty way of getting rid of negatives
def no_neg(x):
    return np.abs(x)
b_centers = np.asarray(list(map(no_neg, b_centers)))
p_centers = np.asarray(list(map(no_neg, p_centers)))

### Getting Samples
x = dt.datetime.now()
b_samples = map(lambda x: np.random.multivariate_normal(x, sigma_osc, size=[n_osc, ]), b_centers)
p_samples = map(lambda x: np.random.multivariate_normal(x, sigma_osc, size=[n_osc, ]), p_centers)
print(dt.datetime.now()-x)

### Converting and Pickling
x = dt.datetime.now()
b_data = list(b_samples)
pkl.dump(b_data, open(files[0], "wb"))
print(dt.datetime.now() - x)
x = dt.datetime.now()
p_data = list(p_samples)
pkl.dump(p_data, open(files[1], "wb"))
print(dt.datetime.now() - x)
