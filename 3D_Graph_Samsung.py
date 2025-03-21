# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:38:44 2019

@author: krish.naik
"""

## Visualization for Multiple Linear Regression

import numpy as np
import pandas as pd
data = pd.read_csv('Samsung Ecosystem_1.csv')
X= data[['DX','MX']]
Y= data['Revenue']




## Prepare the Dataset

df2=pd.DataFrame(X,columns=['DX','MX'])
df2['Revenue']=pd.Series(Y)
df2


## Apply multiple Linear Regression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='Revenue ~ DX + MX', data=df2)
results_formula = model.fit()
results_formula.params



## Prepare the data for Visualizatio
x_surf, y_surf = np.meshgrid(np.linspace(df2.DX.min(), df2.DX.max(), 100),np.linspace(df2.MX.min(), df2.MX.max(), 100))
onlyX = pd.DataFrame({'DX': x_surf.ravel(), 'MX': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



## convert the predicted result in an array
fittedY=np.array(fittedY)




# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['DX'],df2['MX'],df2['Revenue'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('DX')
ax.set_ylabel('MX')
ax.set_zlabel('Revenue')
plt.show()
