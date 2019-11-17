# -*- coding: utf-8 -*-

import Stat
import MatrixHandler
import math
from numpy import linalg as LA

#Find zeros of function with bhaskara
def bhaskara(a, b, c):
    
    delta = math.sqrt( math.pow(b,2) - (4*a*c))
    sub = 2*a
    return [ ((-b + delta) / sub)  ,
            ((-b - delta) / sub)]

#Find covariance between two vectors
def cov(x, y):
    if len(x) != len(y):
        raise Exception('Vetores nao tem o mesmo tamanho')
    
    #Subtracting mean from vectors
    mean_x = Stat.mean(x)
    mean_y = Stat.mean(y)    
    
    a = [x[i]-mean_x for i in range(len(x))]
    b = [y[i]-mean_y for i in range(len(y))]
    
    return sum([a[i]*b[i] for i in range(len(a))])/(len(x)-1)

def matrix_cov(mat):
    #Get number of columns in matrix
    cols = len(mat[0])
    #Calculate covariance matrix
    return [[ cov(MatrixHandler.getColumn(mat,c) , MatrixHandler.getColumn(mat,r)) for r in range(cols)] for c in range(cols)]

def PCA(ds):
    
    #Calculate covariance matrix
    cov = matrix_cov(ds)
    
    #If database has 2 features calculate eigen vectors else call numpy
    if len(ds[0]) == 2:
    
        #Setting bhaskara values
        a = 1
        b = - cov[0][0] + (- cov[1][1])
        c =  (cov[0][0] *  cov[1][1] ) - (cov[0][1] * cov[1][0])
        
        #Finding eigenvalues with baskhara
        eigen_values = bhaskara(a, b, c)
        
        #Finding eigenvectors
        if (cov[1][0] == 0) and (cov[0][1] == 0):
            eigen_vectors = [[1, 0] , [0, 1]]
        elif cov[1][0] != 0:
            eigen_vectors = [[eigen_values[0] - cov[1][1], cov[1][0]] , [eigen_values[1] - cov[1][1] ,cov[1][0]]]
        elif cov[0][1] != 0:
            eigen_vectors = [[cov[0][1] , eigen_values[0] - cov[0][0]], [cov[0][1] , eigen_values[1] - cov[0][0]]]
        else:
            eigen_vectors = None
            
        #Normalizing Eigenvectors with size 1
        norm_eigen = []
        for vec in eigen_vectors:
           norm = math.sqrt( math.pow(vec[0],2) + math.pow(vec[1],2))
           norm_eigen.append([ -(vec[1] / norm), (vec[0] / norm) ])
           
    else:
        norm_eigen = LA.eig(cov)[1].tolist()
        
        
    return norm_eigen