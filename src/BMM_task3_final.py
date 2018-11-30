"""
Created on Mon Nov 21
@author: Aravamuthan Lakshminarayanan
"""

import numpy as np;
from scipy.special import factorial



thetha_initial = np.array([0.6, 0.5]);

def nCr(n,r):
    f = factorial;
    return f(n) / f(r) / f(n-r);

class BMM:

    def __init__(self, k=3, eps=0.00001):
        self.k = k  ## number of clusters
        self.eps = eps  ## threshold to stop `epsilon`

        # All parameters from fitting/learning are kept in a named tuple
        from collections import namedtuple

    def fit_EM(self, X, max_iters=10000):

        # n = number of data-points, d = dimension of data points
        n, d = X.shape

        # randomly choose the starting centroids/means

        thetha = np.array([0.6, 0.5]);

        # initialize the probabilities/weights for each coin
        w = [1. / self.k] * self.k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k coins
        R = np.zeros((n, self.k))

        ### log_likelihoods
        log_likelihoods = []

        P = lambda thetha: nCr(d, np.sum(X, axis=1)) * np.power(thetha, np.sum(X, axis=1)) * np.power((1-thetha), np.subtract(d, np.sum(X, axis=1)));
        # Iterate till max_iters iterations
        while len(log_likelihoods) < max_iters:

            # E - Step

            ## Vectorized implementation of e-step equation to calculate the
            ## membership for each of k -coin
            for k in range(self.k):
                R[:, k] = w[k] * P(thetha[k])

            ### Likelihood computation
            log_likelihood = np.sum(np.log(np.sum(R, axis=1)))

            log_likelihoods.append(log_likelihood)

            ## Normalize so that the responsibility matrix is row stochastic
            R = (R.T / np.sum(R, axis=1)).T

            ## The number of datapoints belonging to each gaussian
            N_ks = np.sum(R, axis=0)

            # M Step
            ## calculate the new thetha for each Coin
            ## utilizing the new responsibilities
            for k in range(self.k):
                ## means
                thetha[k] = 1. / N_ks[k] * np.sum(R[:, k] * (np.sum(X, axis=1)/X.shape[1]));

                ## and finally the probabilities
                w[k] = 1. / n * N_ks[k]
            # check for onvergence
            if len(log_likelihoods) < 2: continue
            if np.abs(log_likelihood - log_likelihoods[-2]) < self.eps: break

        ## bind all results together
        from collections import namedtuple
        self.params = namedtuple('params', ['thetha', 'w', 'log_likelihoods', 'num_iters'])
        self.params.thetha = thetha
        self.params.w = w
        self.params.log_likelihoods = log_likelihoods
        self.params.num_iters = len(log_likelihoods)
        return self.params

    def plot_log_likelihood(self):
        import pylab as plt
        plt.plot(self.params.log_likelihoods)
        plt.title('Log Likelihood vs iteration plot with priors of heads of coins '  + str(thetha_initial))
        plt.xlabel('Iterations')
        plt.ylabel('log likelihood')
        plt.show()




if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filepath", help="File path for data")
    parser.add_option("-k", "--clusters", dest="clusters", help="No. of gaussians")
    parser.add_option("-e", "--eps", dest="epsilon", help="Epsilon to stop")
    parser.add_option("-m", "--maxiters", dest="max_iters", help="Maximum no. of iteration")
    options, args = parser.parse_args()

    #if not options.filepath: raise ('File not provided')

    if not options.clusters:
        print("Used default number of clusters = 3")
        k = 2
    else:
        k = int(options.clusters)

    if not options.epsilon:
        print("Used default eps = 0.0001")
        eps = 0.0001
    else:
        eps = float(options.epsilon)

    if not options.max_iters:
        print("Used default maxiters = 1000")
        max_iters = 1000
    else:
        eps = int(options.maxiters)

    X = np.genfromtxt("Task_3_flip.txt", delimiter=',')
    bmm = BMM(k, eps)
    params = bmm.fit_EM(X, max_iters)
    bmm.plot_log_likelihood()
    print(params.log_likelihoods)
    print("Estimated probabilities are " + str(params.thetha))

    #gmm.plot_log_likelihood()
    # print(gmm.predict(np.array([1, 2])))