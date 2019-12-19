# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import MatrixHandler as mh
import Regressor
import PCA
import LDA
import KMeans
import Perceptron
from tensorflow import keras


root = '../Bases/'
#ds = pd.read_excel(root + 'ShoeSize.xlsx').values.tolist()
#ds = pd.read_excel(root + 'temperature.xlsx').values.tolist()
#ds = pd.read_excel(root + 'alpswater.xlsx').values.tolist()
#ds = pd.read_excel(root + 'US_Census.xlsx').values.tolist()
#ds = pd.read_excel(root + 'Books.xlsx').values.tolist()
#ds = pd.read_excel(root + 'PCA_Example.xlsx').values.tolist()
#ds = pd.read_excel(root + 'KMeans.xlsx').values
#ds = pd.read_excel(root + 'Wholesale.xlsx').values
#ds = pd.read_excel(root + 'user_model.xlsx').values
#ds = pd.read_excel(root + 'and.xlsx').values
#ds = pd.read_excel(root + 'or.xlsx').values
#ds = pd.read_excel(root + 'xor.xlsx').values


#data = datasets.load_iris()
#ds = pd.DataFrame(data['data'], columns=data['feature_names'])
#ds['target'] = data['target']
#ds = ds.values

# =============================================================================
# #Exercicio 1 - Regressao Linear
# =============================================================================

#x = [ds.iloc[:,0].tolist()]
#y = [ds.iloc[:,1].tolist()]
#betas = Regressor.getBetas(x, y)
#x_linha = [min(x[1]), max(x[1])]
#y_linha = Regressor.predict(x_linha, betas)
#plt.plot(x[1], y[0], '*', color='blue')
#plt.plot(x_linha, y_linha, '-', color='black')
#
#x = [ds.iloc[:,0].tolist()]
#y = [ds.iloc[:,1].tolist()]
#betas_q = Regressor.getBetasQuadratic(x, y)
#x_linha = [np.linspace(min(x[1]), max(x[1]), num=100)]
#y_linha = Regressor.predictSquare(x_linha, betas_q)
#plt.plot(x_linha, y_linha, '.', color='red')
#
#x = [ds.iloc[:,0].tolist()]
#y = [ds.iloc[:,1].tolist()]
#betas = Regressor.getBetasRobustic(x, y)
#x_linha = [min(x[1]), max(x[1])]
#y_linha = Regressor.predict(x_linha, betas)
#plt.plot(x_linha, y_linha, '-', color='green')
#
#plt.show()

# =============================================================================
# #Exercicio 2 - PCA
# =============================================================================

#eigen_vectors = PCA.PCA(ds)
#
#x = mh.getColumn(ds, 0)
#y = mh.getColumn(ds, 1)
#z = mh.getColumn(ds, 2)
#pc1 = x.copy()
#pc2 = x.copy()
#
#for i in range(len(x)):
#    pc1[i] = (x[i] *  eigen_vectors[0][0]) + (y[i] * eigen_vectors[0][1]) + (z[i] * eigen_vectors[0][2])
#
#for i in range(len(x)):
#    pc2[i] = (x[i] *  eigen_vectors[2][0]) + (y[i] * eigen_vectors[2][1]) + (z[i] * eigen_vectors[2][2])
#
#
#plt.scatter(x, y)
#plt.show()
#
#plt.scatter(pc1, pc2)
#plt.show()
#
#fig = plt.figure()
#ax = Axes3D(fig)
#x = mh.getColumn(ds, 0)
#y = mh.getColumn(ds, 1)
#z = mh.getColumn(ds, 2)
#fig = plt.figure()
#ax.scatter(x, y, z)
#plt.show()


# =============================================================================
# #Exercicio 3 - LDA
# =============================================================================

#lda_data = LDA.fit(ds)
#
#x=lda_data[:, 0]
#y=lda_data[:, 1]
#classes = lda_data[:, 2]
#unique = list(set(classes))
#colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
#for i, u in enumerate(unique):
#    xi = [x[j] for j  in range(len(x)) if classes[j] == u]
#    yi = [y[j] for j  in range(len(x)) if classes[j] == u]
#    plt.scatter(xi, yi, c=colors[i], label=str(u))
#plt.legend()
#
#plt.show()

# =============================================================================
# #Exercicio 4 - KMeans
# =============================================================================


#centers = [(-5, -5), (5, 5), (-4,4)]
#cluster_std = [1, 1.2, 1.0]
#
#X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=4, random_state=1)
#
#new_data, centroids = KMeans.KMeans(X, 4)
#
#plt.scatter(new_data[:, 0], new_data[:, 1], c=new_data[:, 2])
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
#plt.show()

# =============================================================================
# #Exercicio 5 - Perceptron
# =============================================================================

#weights = Perceptron.fit(ds, 50, 0.1)
#print(Perceptron.predict([0,1], weights))


# =============================================================================
# #Exercicio 6 - Multi Layer Perceptrons
# =============================================================================

# =============================================================================
# #Exercicio 7 - Convolutional Neural Networks
# =============================================================================


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#Define uma rede neural sequencial
lenet = keras.Sequential()
#Define uma camada de convolução 
lenet.add(keras.layers.Conv1D(filters=6, 
                        kernel_size=3, 
                        activation='relu', 
                        use_bias=True,
                        input_shape=(28,28)))
#Define uma camada de pooling 
lenet.add(keras.layers.MaxPooling1D(strides=2,
                                    pool_size=2))
#Define uma camada de convolução 
lenet.add(keras.layers.Conv1D(filters=16, 
                        kernel_size=3, 
                        activation='relu'))
#Define uma camada de pooling
lenet.add(keras.layers.MaxPooling1D(strides=2,
                                    pool_size=2))
#Transformação de matrizes em um vetor de entrada para uma MLP
lenet.add(keras.layers.Flatten())
#Define uma camada de uma MLP com 120 neurônios
lenet.add(keras.layers.Dense(units=120, activation='relu'))
#Define uma camada de uma MLP com 84 neurônios
lenet.add(keras.layers.Dense(units=84, activation='relu'))
#Define uma camada de uma MLP com 10 neurônios
lenet.add(keras.layers.Dense(units=10, activation = 'softmax'))

#Definições de hiper parâmetro
lenet.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

lenet.fit(x_train, y_train, epochs=10, batch_size=128)

score = lenet.evaluate(x_test, y_test, verbose=0)
print('Final loss:', score[0])
print('Final acc:', score[1])















