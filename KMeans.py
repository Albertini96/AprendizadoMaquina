# -*- coding: utf-8 -*-

from random import randrange
import MatrixHandler
import matplotlib.pyplot as plt
import numpy as np
import math
import Stat
def calculate_distance(a, b):
    return math.sqrt(sum(np.square(a-b)))

def NDMean(matrix):
    return np.sum(matrix, axis=0)/len(matrix)

def get_clusters(centroids, k, ds):
    cluster = []
    for sample in ds:
        select_cluster = 0
        dist = 999999999999
        for centroid in range(k):
            if calculate_distance(sample, centroids[centroid]) < dist:
                dist = calculate_distance(sample, centroids[centroid])
                select_cluster = centroid 
        cluster.append(select_cluster)
    return cluster

def KMeans(ds, k):  
    
    number_centroids = k
    
    x = MatrixHandler.transposeMatrix(ds)
    
    random_centroids = []
    #For each centroids
    for i in range(0, number_centroids):
        random_sample = randrange(len(x[0]))    
        dim = []
        #For each dimension of X
        for j in range(len(x)):
            dim.append(x[j][random_sample])                
        random_centroids.append(dim)
    random_centroids = np.asarray(random_centroids)
    
    #Get clusters
    cluster = get_clusters(random_centroids, k, ds)
    
#    t=x[0]
#    y=x[1]
#    classes = cluster
#    unique = list(set(classes))
#    colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
#    for i, u in enumerate(unique):
#        xi = [t[j] for j  in range(len(t)) if classes[j] == u]
#        yi = [y[j] for j  in range(len(t)) if classes[j] == u]
#        plt.scatter(xi, yi, c=colors[i], label=str(u))
#    plt.legend()
#    random_centroids = MatrixHandler.transposeMatrix(random_centroids)
#    plt.scatter(random_centroids[0], random_centroids[1], color='red')
#    plt.show()
    
    last_centroids = random_centroids
    
    is_eq_last = False
    
    while(not is_eq_last):
        new_centroids = []

        for c in Stat.unique(cluster):
            new_centroids.append(NDMean(ds[np.asarray(cluster) == c]).tolist())
        new_centroids = np.asarray(new_centroids)
        is_eq_last = np.array_equal(new_centroids,last_centroids)
        
        last_centroids = new_centroids
        
        cluster = get_clusters(new_centroids, k, ds)
        
#        t=x[0]
#        y=x[1]
#        classes = cluster
#        unique = list(set(classes))
#        colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
#        for i, u in enumerate(unique):
#            xi = [t[j] for j  in range(len(t)) if classes[j] == u]
#            yi = [y[j] for j  in range(len(t)) if classes[j] == u]
#            plt.scatter(xi, yi, c=colors[i], label=str(u))
#        plt.legend()
#        new_centroids = MatrixHandler.transposeMatrix(new_centroids)
#        plt.scatter(new_centroids[0], new_centroids[1], color='red')
#        plt.show()
        
#    print(ds)
#    print(cluster)

    return np.append(ds, np.asarray(cluster).reshape(len(cluster), 1), axis=1), last_centroids

    
 