from scipy.stats import multivariate_normal;
import numpy as np;
import random;


# def get_density(x, mu, stdev):
#     mu = np.array(mu.T);
#     stdev = np.array(stdev.T);
#     mvn = multivariate_normal(mu, stdev);
#     return mvn.pdf(np.array(x.T));


def get_density(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = (np.dot(np.dot((pos-mu).T, Sigma_inv), pos-mu));

    return float(np.exp(-fac / 2) / N);


def calculate_the_datavariane(fileName):
    x = np.zeros((2,1600));
    with open(fileName) as f:
        i = 0;
        for l in f:
            data = np.zeros((2,1));
            data[0,0] = (l.strip().split(" ")[0]);
            data[1,0] = (l.strip().split(" ")[1]);
            x[0,i] = data[0,0];
            x[1,i] = data[1,0];
            i+=1;
    return np.cov(x), x;



def get_posterior(x, priors, means, stdevs):
    posteriors = [];

    for i in range(len(priors)):
        posteriors.append(get_density(x, means[i], stdevs[i]) * priors[i]);
    sum_posteriors = sum(posteriors);
    for i in range(len(priors)):
        posteriors[i] /= sum_posteriors;
    return posteriors;

def main():

    #no_of_gaussion = int(sys.argv[1]);
    no_of_gaussion = 2;

    #fileName = sys.argv[2];
    priors = [];
    stdves = [];
    means = [];
    initail_covariance, data = calculate_the_datavariane("DS_1.txt");
    for i in range(no_of_gaussion):
        prior = 1/no_of_gaussion;
        priors.append(prior);
        mean = np.zeros((2, 1));
        means.append(mean);
        stdves.append(initail_covariance*random.uniform(0, 1));

    for i in range(data.shape[1]):
        plines = [];
        plines.append(get_posterior(data[:,[i]], priors, means, stdves));

        (priors, means, stdves) = relearn(plines);

        print(get_posterior(data[:,[i]], priors, means, stdves));

main();


