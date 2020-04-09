import matplotlib.pyplot as plt
import numpy as np
import config
data_fig_dir = config.data_fig_dir

######################################################################################
##################################### Data_generation ################################
######################################################################################
def shuffle(X, y):
    y = np.array(y)
    idx = np.random.permutation(y.shape[0])
    return X[idx], y[idx]

def no_neg(x):
    return np.abs(x)

def unpack_sample(s):
    data = list(s)
    data = [i.flatten(order = 'F') for i in data]
    data = np.asarray(data)
    return data

def samples_2_data(ldf, lbls, ratio):
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(ldf)):
        n = ldf[i].shape[0]
        cut = int(ratio*n)
        if len(y_train) == 0:
            X_train = ldf[i][:cut,]
            X_test = ldf[i][cut:,]
            y_train = [lbls[i]]*cut
            y_test = [lbls[i]]*(n-cut)
        else:
            X_train = np.vstack([X_train, ldf[i][:cut,]])
            X_test = np.vstack([X_test, ldf[i][cut:,]])
            y_train.extend([lbls[i]]*cut)
            y_test.extend([lbls[i]]*(n-cut))
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    return X_train, y_train, X_test, y_test

def simData(n_sub, n_samples, mu_b, mu_p, sigma_b, sigma_p, sigma_sub, train_ratio = 0.75, file = None):
    ### Distribution Samples
    b_centers = np.random.multivariate_normal(mu_b, sigma_b, size=[n_samples,])
    p_centers = np.random.multivariate_normal(mu_p, sigma_p, size=[n_samples,])
    ### Quick'n'Dirty way of getting rid of negatives
    b_centers = np.asarray(list(map(no_neg, b_centers)))
    p_centers = np.asarray(list(map(no_neg, p_centers)))
    ### Getting Samples
    b_samples = map(lambda x: np.random.multivariate_normal(x, sigma_sub, size=[n_sub, ]), b_centers)
    p_samples = map(lambda x: np.random.multivariate_normal(x, sigma_sub, size=[n_sub, ]), p_centers)
    ### Unpacking Samples
    b_data = unpack_sample(b_samples)
    p_data = unpack_sample(p_samples)
    ###also generate plots

    # print(b_centers[0:3, 0], b_centers[0:3, 1])
    # plt.scatter(b_data[:, 0:15], b_data[:, 15:30])
    # plt.scatter(p_data[:, 0:15], p_data[:, 15:30])
    plt.scatter(b_centers[:, 0], b_centers[:, 1])
    plt.scatter(p_centers[:, 0], p_centers[:, 1])
    plt.xlim((0,1))
    #plt.title()
    if type(file) == str:
        splfile = file.split('/')[-1]
        figname = data_fig_dir + splfile.split('.')[0] + '.png'
        print(figname)
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
        plt.close()
    ### Convert Samples to Dataframes
    X_tr, y_tr, X_te, y_te = samples_2_data([b_data, p_data], [1, 0], train_ratio)
    return X_tr, y_tr, X_te,y_te

def simTestData(n_sub, sigma_sub, file = None):
    b = np.array([0.5, 0.2])
    p = np.array([0.5, 0.8])

    b_samples = np.random.multivariate_normal(b, sigma_sub, size=[n_sub, ])
    p_samples = np.random.multivariate_normal(p, sigma_sub, size=[n_sub, ])

    b_data = b_samples.flatten(order = 'F')
    p_data = p_samples.flatten(order = 'F')


    plt.scatter(b_data[0:15], b_data[15:30])
    plt.scatter(p_data[0:15], p_data[15:30])
    plt.xlim((0,1))
    #plt.title()
    if type(file) == str:
        splfile = file.split('/')[-1]
        figname = data_fig_dir + splfile.split('.')[0] + '.png'
        print(figname)
        plt.savefig(figname)
        plt.close()
    else:
        plt.show()
        plt.close()
    ### Convert Samples to Dataframes

    return [b_data], 1, [p_data], 0

######################################################################################
################################## Exposure learning #################################
######################################################################################
