# -*- coding: utf-8 -*-

import MatrixHandler

#Return the betas that will be used as cofactors on regression
def getBetas(x, y):
    
    #Including column of 1 on X matrix
    x.insert(0,([1 for i in range(len(x[0]))]))

    #Transposing X
    transposed_x = MatrixHandler.transposeMatrix(x)
    
    #Calculating betas
    mult = MatrixHandler.multMatrix(transposed_x, x)
    mult_inv = MatrixHandler.getMatrixInverse(mult)
    X_y = MatrixHandler.multMatrix(transposed_x, y)
    betas = MatrixHandler.multMatrix(mult_inv, X_y)
    
    return betas

#Return the betas that will be used as cofactors on quadratic regression
def getBetasQuadratic(x, y):
    
    #Including column of 1 on X matrix
    x.insert(0,([1 for i in range(len(x[0]))]))

    #Including column of quadratic X on X matrix
    for j in range(1, len(x)):
        x.append([x[j][i]**2 for i in range(len(x[0]))])
        
    #Transposing X
    transposed_x = MatrixHandler.transposeMatrix(x)
    
    #Calculating betas
    mult = MatrixHandler.multMatrix(transposed_x, x)
    mult_inv = MatrixHandler.getMatrixInverse(mult)
    X_y = MatrixHandler.multMatrix(transposed_x, y)
    betas = MatrixHandler.multMatrix(mult_inv, X_y)
    
    return betas

#Return the betas that will be used as cofactors on robustic regression
def getBetasRobustic(x, y):
    
    #Getting copies of X and Y
    x_ = x.copy()
    y_ = y.copy()
    
    #Running linear regression
    betas_linear = getBetas(x, y)
    
    #Predicting X's with linear regression
    y_lin_predicted = predict(x[1], betas_linear)

    #Finding Weights
    w = [abs((1/(y[0][i] - y_lin_predicted[i]))) for i in range(len(y_lin_predicted))]
    
    #Inserting column of 1 in X Matrix
    x_.insert(0,([1 for i in range(len(x[0]))]))
    
    #Transposing X
    transposed_x = MatrixHandler.transposeMatrix(x_)
    transposed_x_ = MatrixHandler.transposeMatrix(x_)
    
    #Performing scalar multiplication on X
    for i in range(len(transposed_x)):
        for j in range(len(transposed_x[0])):
            transposed_x[i][j] = transposed_x[i][j] * w[i]

    #Performing scalar multiplication on Y
    for i in range(len(y_[0])):
        y_[0][i] = y_[0][i] * w[i]

    #Calculating betas
    mult = MatrixHandler.multMatrix(transposed_x, x_)    
    mult_inv = MatrixHandler.getMatrixInverse(mult)
    X_y = MatrixHandler.multMatrix(transposed_x_, y_)
    betas = MatrixHandler.multMatrix(mult_inv, X_y)
    
    return betas

#Predict future values based on new X and betas for linear and robustic regression
def predict(x, betas):
    if type(x) == list:
        return [i * betas[0][1] + betas[0][0] for i in x]
    else:
        return x * betas[0][1] + betas[0][0]

#Predict future values based on new X and betas for quadratic regression    
def predictSquare(x, betas):
    if type(x) == list:
        return [(i**2 * betas[0][2])+ (i * betas[0][1]) + betas[0][0] for i in x]
    else:
        return (x**2 * betas[0][2]) + (x * betas[0][1]) + betas[0][0]