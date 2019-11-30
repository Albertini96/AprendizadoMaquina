# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import MatrixHandler as mh
import Regressor
import PCA
import LDA
import KMeans

root = '../Bases/'
#ds = pd.read_excel(root + 'ShoeSize.xlsx').values.tolist()
#ds = pd.read_excel(root + 'temperature.xlsx').values.tolist()
#ds = pd.read_excel(root + 'alpswater.xlsx').values.tolist()
#ds = pd.read_excel(root + 'US_Census.xlsx').values.tolist()
#ds = pd.read_excel(root + 'Books.xlsx').values.tolist()
#ds = pd.read_excel(root + 'PCA_Example.xlsx').values.tolist()

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

lda_data = LDA.fit(ds)

x=lda_data[:, 0]
y=lda_data[:, 1]
classes = lda_data[:, 2]
unique = list(set(classes))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x[j] for j  in range(len(x)) if classes[j] == u]
    yi = [y[j] for j  in range(len(x)) if classes[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()

plt.show()

# =============================================================================
# #Exercicio 4 - KMeans
# =============================================================================

#centroids = KMeans.KMeans(2)













