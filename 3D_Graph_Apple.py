# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:38:44 2019

@author: krish.naik
"""

## Visualization for Multiple Linear Regression

import numpy as np
import pandas as pd
data = pd.read_csv('Apple_Ecosystem.csv')
X= data[['Products','Services']]
Y= data['Reveneue']




## Prepare the Dataset

df2=pd.DataFrame(X,columns=['Products','Services'])
df2['Reveneue']=pd.Series(Y)
df2


## Apply multiple Linear Regression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='Reveneue ~ Products + Services', data=df2)
results_formula = model.fit()
results_formula.params



## Prepare the data for Visualizatio
x_surf, y_surf = np.meshgrid(np.linspace(df2.Products.min(), df2.Products.max(), 100),np.linspace(df2.Services.min(), df2.Services.max(), 100))
onlyX = pd.DataFrame({'Products': x_surf.ravel(), 'Services': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



## convert the predicted result in an array
fittedY=np.array(fittedY)




# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Products'],df2['Services'],df2['Reveneue'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Products')
ax.set_ylabel('Services')
ax.set_zlabel('Reveneue')
plt.show()
