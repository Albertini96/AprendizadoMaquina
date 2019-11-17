# -*- coding: utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from random import randrange
import MatrixHandler
import matplotlib.pyplot as plt

def KMeans(k):    
    centers = [(-5, -5), (5, 5)]
    cluster_std = [0.8, 1]
    
    X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
    
    
    number_centroids = k
    
    
    x = MatrixHandler.transposeMatrix(X)
    random_centroids = []
    
    #For each centroids
    for i in range(0, number_centroids):
        random_sample = randrange(len(x[0]))
        
        dim = []
        #For each dimension of X
        for j in range(len(x)):
            dim.append(x[j][random_sample])
            
            
        random_centroids.append(dim)
        
    
    
    random_centroids = MatrixHandler.transposeMatrix(random_centroids)
    random_centroids
    
    plt.scatter(x[0], x[1])
    plt.scatter(random_centroids[0], random_centroids[1], color='red')