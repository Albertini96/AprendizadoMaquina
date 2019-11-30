# -*- coding: utf-8 -*-

import Stat
import MatrixHandler as mh
import numpy as np

#Calcula a média de cada coluna da base de dados caso não seja fornecido o segundo argumento
#Se o segundo argumento for fornecido a função calcula a matriz de média por caracteristicas e por classes
def grand_mean(ds, column= -1):
    
    if column > -1:
        classes = Stat.unique(mh.getColumn(ds, len(ds[0])-1))
        
        mean = []

        for c in classes:
            transposta = mh.transposeMatrix(ds.copy())            
            transposta = np.array(transposta)            
            normal = ds[transposta[len(ds[0])-1] == c]            
            mean.append([Stat.mean(mh.getColumn(normal, i)) for i in range(0, len(normal[0]) - 1)])
            
        return mean
    else:
        return [Stat.mean(mh.getColumn(ds, i)) for i in range(0, len(ds[0]) - 1)]

#Calcula a variância de cada classe
def scatter_within_class(ds, class_mean):
    classes = Stat.unique(mh.getColumn(ds, len(ds[0])-1))
    sw = np.zeros((len(ds[0]) - 1, len(ds[0]) - 1))
    
    for i in range(len(classes)):
        si = np.zeros((len(ds[0]) - 1, len(ds[0]) - 1))
        
        transposta = mh.transposeMatrix(ds.copy())            
        transposta = np.array(transposta)            
        normal = ds[transposta[len(ds[0])-1] == classes[i]]     

        for j in range(len(normal)):
            row = normal[j]
            row = row[0:-1].reshape(4,1)
            mean = np.asarray(class_mean[i]).reshape(4,1)

            si += (row - mean).dot((row-mean).T)
        sw += si
        
    return sw
    
#Calcula a variância entre classes
def scatter_between_class(ds, class_mean, grand_mean):
    classes = Stat.unique(mh.getColumn(ds, len(ds[0])-1))
    sb = np.zeros((len(ds[0]) - 1, len(ds[0]) - 1))
    
    for i in range(len(class_mean)):
        transposta = mh.transposeMatrix(ds.copy())            
        transposta = np.array(transposta)            
        normal = ds[transposta[len(ds[0])-1] == classes[i]]    
        
        meanc = np.asarray(class_mean[i]).reshape(4,1)
        meang = np.asarray(grand_mean).reshape(4,1)
        sb += len(normal) * (meanc - meang).dot((meanc - meang).T)
    return sb
    
#Faz o calculo dos novos dados tranformados pelo algoritmo LDA
def transform_data(data, eigen_vectors, dimensions=2):
    
    data_ = data[:,:-1]

    labels = data[:, len(data[0]) - 1]

    eig = eigen_vectors[:, 0:dimensions]

    transformed_data = np.dot(data_, eig)
    
    transformed_data = np.hstack((transformed_data, np.atleast_2d(labels).T)) 

    return transformed_data
    
#Computa os novos eixos da base de dados
def fit(ds):
    
    sw = scatter_within_class(ds, grand_mean(ds, len(ds[0])-1))

    sb = scatter_between_class(ds, grand_mean(ds, len(ds[0])-1), grand_mean(ds))

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sw).dot(sb))
    
    new_data = transform_data(ds, eig_vecs, 2)
    
    return new_data