import numpy as np


###Define paths

root_dir = '/Users/charleswu/Google Drive/HoltLab/CueWeighting/Modeling/Network/NeuralSpeechAdaptation/'
data_dir = root_dir + 'Data/'

pre_file = data_dir + 'pretrain.pkl'
can_file = data_dir + 'canonical.pkl'
rev_file = data_dir + 'rev.pkl'
test_file = data_dir + 'test.pkl'
test_file_hor = data_dir + 'test_hor.pkl'

train_fig_dir = root_dir + 'Figure_training/'
data_fig_dir = root_dir + 'Figure_data/'

######################################################################################
################################### Data generation ##################################
######################################################################################
###key parameters for the centers and variances of the two distributions for b and p
### Sampling Sizes

n_sub = 15 ### number of inputs units in each bank of input
n_samples = 10000 ### number of samples sampled from each distribution for pretraining
#n_acc_samples = 250 ###
n_exposure = 50

###Pre-training and canonical
mu_b = np.array([0.2, 0.2])
mu_p = np.array([0.8, 0.8])

# sigma_b = np.array([[0.006, 0], [0, 0.016]])
# sigma_p = np.array([[0.006, 0], [0, 0.016]])

sigma_b = np.array([[0.02, 0], [0, 0.08]])
sigma_p = np.array([[0.02, 0], [0, 0.08]])

sigma_b_exposure = np.array([[0.006, 0], [0, 0.016]])
sigma_p_exposure = np.array([[0.006, 0], [0, 0.016]])

sigma_sub = np.array([[0.00001, 0], [0, 0.00001]])

###Reverse
mu_rev_b = np.array([0.2, 0.8])
mu_rev_p = np.array([0.8, 0.2])

###Test 
mu_low = np.array([0.5, 0.2])
mu_high = np.array([0.5, 0.8])

sigma_low = np.array([[0, 0], [0, 0]])
sigma_high = np.array([[0, 0], [0, 0]])


######################################################################################
###################################### Pre-training ##################################
######################################################################################
epochs = 1
batch_size = 10