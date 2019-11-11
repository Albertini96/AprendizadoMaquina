# -*- coding: utf-8 -*-


def mean(vec):
    return sum(vec)/len(vec)


def cov(x, y):
    if len(x) != len(y):
        raise Exception('Vetores nao tem o mesmo tamanho')
    
    mean_x = mean(x)
    mean_y = mean(y)    
    
    a = [x[i]-mean_x for i in range(len(x))]
    b = [y[i]-mean_y for i in range(len(y))]
    
    return sum([a[i]*b[i] for i in range(len(a))])/(len(x)-1)

def matrix_cov(mat):
    return [[cov(mat[i], mat[j]) for i in range(len(mat))] for j in range(len(mat))]