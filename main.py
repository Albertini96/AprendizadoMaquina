# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import Regressor
import PCA
import KMeans

root = '../Bases/'
#ds = pd.read_excel(root + 'ShoeSize.xlsx').values.tolist()
#ds = pd.read_excel(root + 'temperature.xlsx').values.tolist()
ds = pd.read_excel(root + 'alpswater.xlsx').values.tolist()
#ds = pd.read_excel(root + 'US_Census.xlsx').values.tolist()
#ds = pd.read_excel(root + 'Books.xlsx').values.tolist()
#ds = pd.read_excel(root + 'PCA_Example.xlsx').values.tolist()


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

eigen_vectors = PCA.PCA(ds)

# =============================================================================
# #Exercicio 3 - KMeans
# =============================================================================

#centroids = KMeans.KMeans(2)













