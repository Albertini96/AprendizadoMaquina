# -*- coding: utf-8 -*-

import MatrixHandler
import Stat
import pandas as pd
import Regressor
import matplotlib.pyplot as plt
import numpy as np

ds = pd.read_excel('ShoeSize.xlsx')
#ds = pd.read_excel('temperature.xlsx')
#ds = pd.read_excel('alpswater.xlsx')
#ds = pd.read_excel('US_Census.xlsx')
#ds = pd.read_excel('Books.xlsx')

x = [ds.iloc[:,0].tolist()]
y = [ds.iloc[:,1].tolist()]
betas = Regressor.getBetas(x, y)
x_linha = [min(x[1]), max(x[1])]
y_linha = Regressor.predict(x_linha, betas)
plt.plot(x[1], y[0], '*', color='blue')
plt.plot(x_linha, y_linha, '-', color='black')

x = [ds.iloc[:,0].tolist()]
y = [ds.iloc[:,1].tolist()]
betas_q = Regressor.getBetasQuadratic(x, y)
x_linha = [np.linspace(min(x[1]), max(x[1]), num=100)]
y_linha = Regressor.predictSquare(x_linha, betas_q)
plt.plot(x_linha, y_linha, '.', color='red')

x = [ds.iloc[:,0].tolist()]
y = [ds.iloc[:,1].tolist()]
betas = Regressor.getBetasRobustic(x, y)
x_linha = [min(x[1]), max(x[1])]
y_linha = Regressor.predict(x_linha, betas)
plt.plot(x_linha, y_linha, '-', color='green')

plt.show()



