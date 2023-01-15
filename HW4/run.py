"""
EECS 445 - Introduction to Machine Learning
Winter 2019 - HW4 - run.py
Script for running GMM soft clustering
"""

from sklearn.datasets import load_iris
from scipy.misc import imresize

import matplotlib.pyplot as plt
import numpy as np
import string as s

from gmm import gmm


def get_data():
    X = load_iris().data
    print("Shape of the input data: %d by %d" % (X.shape[0], X.shape[1]))
    return X


def main():
    """
    GMM call with different values of number of clusters
        - num_K is an array containing the tested cluster sizes
        - cluster_proportions maps each cluster size to a size by 1 vector containing the mixture proportions
        - means is a dictionary mapping the cluster size to matrix of means
        - z_K maps each cluster size into a num_points by k matrix of pointwise cluster membership probabilities
        - sigma2 maps each cluster size to the corresponding sigma^2 value learnt
        - BIC_K contains the best BIC values for each of the cluster sizes
    """
    print("We'll try different numbers of clusters with GMM, using multiple runs for each to identify the 'best' results")
    trainX = get_data()
    num_K = range(2, 9) # List of cluster sizes
    BIC_K = np.zeros(len(num_K))
    means = {} # Dictionary mapping cluster size to corresponding matrix of means
    cluster_proportions = {} # Dictionary mapping cluster size to corresponding mixture proportions vector
    z_K = {}
    sigma2 = {} # Dictionary mapping cluster size to the learned variance value
    for idx in range(len(num_K)):
        # Running
        k = num_K[idx]
        print("%d clusters..." % k)
        bestBIC = float("inf")
        for i in range(1, 11):
            # TODO: Run gmm function 10 times and get the best
            # set of parameters for this particular value of k
            log_like = gmm(trainX, k)[4]
            if log_like < bestBIC: bestBIC = log_like
        BIC_K[idx] = bestBIC

    # TODO: Part d: Make a plot to show BIC as function of clusters K
    plt.plot(num_K, BIC_K)
    plt.xlabel("num_cluster")
    plt.ylabel("BestBic_Value")
    plt.show()

if __name__ == "__main__":
    main() 