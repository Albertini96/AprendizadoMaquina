# -*- coding: utf-8 -*-
import numpy as np

#Defines a activation function for perceptron
def activation_function(x):
    return 1 if x>=0 else 0

#Predict based on weights learned
def predict(features, weights):
    
    features = np.append(features, 1)
    
    predicted = 0
    for x in range(len(features)):      
        predicted += features[x] * weights[x]

    return activation_function(predicted)

#Fit perceptron to dataset
#Y to predict must be on last column of dataset
def fit(ds, epochs, lr):
    
    #Initializing weights
    weights = np.ones(ds.shape[1])
    
    #Obtaining only collumns with features
    features = ds[:,0:ds.shape[1]-1]
    
    #Appending bias
    features = np.append(features, np.ones(len(features)).reshape(len(features),1), axis=1)
    
    #Retrieving Y to predict from dataset
    y = ds[:,ds.shape[1]-1]
    
    predicted = None
    
    #For each epoch do
    for epoch in range(epochs):
        print('Weights ->',  weights)
        
        #For each row in dataset do
        for row in range(len(features)):
            predicted = 0
           
            #Multiplying features by weights            
            for x in range(len(features[row])):      
                print('X' + str(x), features[row][x])
                predicted += features[row][x] * weights[x]
             
            #Updating weights
            for w in range(len(weights)):
                weights[w] += lr * (y[row] - activation_function(predicted)) * features[row][w]
                
            print('Real ->',  y[row])
            print('Predicted ->',  activation_function(predicted))
            print('New Weights ->',  weights)
            
            print()
        print('#============================#')
    
    return weights