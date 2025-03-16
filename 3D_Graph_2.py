# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:38:44 2019

@author: krish.naik
"""

## Visualization for Multiple Linear Regression

import numpy as np
import pandas as pd
data = pd.read_csv('Google_Ecosystem_Dataset_1.csv')
X= data[['Advertising','Other']]
Y= data['Revenue']




## Prepare the Dataset

df2=pd.DataFrame(X,columns=['Advertising','Other'])
df2['Revenue']=pd.Series(Y)
df2


## Apply multiple Linear Regression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='Revenue ~ Advertising + Other', data=df2)
results_formula = model.fit()
results_formula.params



## Prepare the data for Visualization

x_surf, y_surf = np.meshgrid(np.linspace(df2.Advertising.min(), df2.Advertising.max(), 100),np.linspace(df2.Other.min(), df2.Other.max(), 100))
onlyX = pd.DataFrame({'Advertising': x_surf.ravel(), 'Other': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



## convert the predicted result in an array
fittedY=np.array(fittedY)




# Visualize the Data for Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Advertising'],df2['Other'],df2['Revenue'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Advertising')
ax.set_ylabel('Other')
ax.set_zlabel('Revenue')
plt.show()
