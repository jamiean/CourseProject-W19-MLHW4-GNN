"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - HW4 - gmm.py
The gmm function takes in as input a data matrix X and a number of gaussians in the mixture model
The implementation assumes that the covariance matrix is shared and is a spherical diagonal covariance matrix
"""

from scipy.stats import norm, multivariate_normal
from scipy.misc import logsumexp
import numpy as np
import math


def calc_logpdf(x, mean, cov):
    x = multivariate_normal.logpdf(x, mean=mean, cov=cov)
    return x


def gmm(trainX, num_K, num_iter = 10):
    """
        input trainX is a N by D matrix containing N datapoints, num_K is the number of clusters or mixture components desired.
        num_iter is the maximum number of EM iterations run over the dataset
        Description of other variables:
            - mu which is K*D, the coordinates of the means
            - pk, which is K*1 and represents the cluster proportions
            - zk, which is N*K, has at each z(n,k) the probability that the nth data point belongs to cluster k, specifying the cluster associated with each data point
            - si2 is the estimated (shared) variance of the data
            - BIC is the Bayesian Information Criterion (smaller BIC is better)
    """
    N = trainX.shape[0]
    D = trainX.shape[1]

    try:
        if num_K >= N:
            raise AssertionError
    except AssertionError:
        print("You are trying too many clusters")
        raise

    si2 = 5 # Initialization of variance
    pk = np.ones((num_K,1))/num_K # Uniformly initialize cluster proportions
    mu = np.random.randn(num_K, D) # Random initialization of clusters
    zk = np.zeros([N,num_K]) # Matrix containing cluster membership probability for each point

    for iter in range(0,num_iter):
        """
            E-Step
            In the first step, we find the expected log-likelihood of the data which is equivalent to:
            finding cluster assignments for each point probabilistically
            In this section, you will calculate the values of zk(n,k) for all n and k according to current values of si2, pk and mu
        """
        # TODO

        for i in range(0, N):
            for j in range(0, num_K):
                zk[i, j] = np.log(pk[j]) + calc_logpdf(trainX[i], mu[j], si2)
            zk[i] = zk[i] - logsumexp(zk[i])


        """
            M-step
            Compute the GMM parameters from the expressions which you have in your writeup
        """

        # Estimate new value of pk
        # TODO
        for j in range(0, num_K): pk[j] = np.exp(logsumexp(zk[:,j])) / N

        # Estimate new value for means
        # TODO
        for j in range(0, num_K): mu[j] = np.matmul([np.exp(zk[:,j])], trainX) / np.exp(logsumexp(zk[:,j]))

        # Estimate new value for sigma^2
        # TODO
        transform = np.zeros([N, num_K])
        for i in range(0, N):
            for j in range(0, num_K):
                transform[i, j] = (trainX[i] - mu[j]).dot(trainX[i] - mu[j])
        si2 = np.exp(logsumexp(zk, b = transform)) / (N * D)

    # Computing the expected likelihood of data for the optimal parameters computed
    # TODO
    for i in range(0, N):
        for j in range(0, num_K):
            zk[i, j] = np.log(pk[j]) + calc_logpdf(trainX[i], mu[j], si2)

    log_like = np.sum(logsumexp(zk, 1))

    # Compute the BIC for the current cluster
    # TODO
    BIC = (D + 1) * num_K * np.log(N) - 2 * log_like

    return mu, pk, zk, si2, BIC
