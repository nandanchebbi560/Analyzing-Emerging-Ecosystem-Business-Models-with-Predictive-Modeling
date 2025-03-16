import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USAhousing = pd.read_csv('Apple_Ecosystem.csv')
USAhousing.head()

X = USAhousing[['Year','Products','Services']]
y = USAhousing['Reveneue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)

    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation for MLR:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation for MLR:\n_____________________________________')
print_evaluate(y_train, train_pred)

#Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

print('Test set evaluation for SDG:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation for SDG:\n_____________________________________')
print_evaluate(y_train, train_pred)


#Polynomial Regression Facing Issues due to Normalize Functions

#ElasticNet

from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation for Elastic Net:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation for Elastic Net:\n_____________________________________')
print_evaluate(y_train, train_pred)

#Lasso Regression
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1,
              precompute=True,
#               warm_start=True,
              positive=True,
              selection='random',
              random_state=42)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation for Lasso:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation for Lasso:\n_____________________________________')
print_evaluate(y_train, train_pred)

#Ridge Regression
from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation for Ridge:\n_____________________________________')
print(y_test, test_pred)
print('====================================')
print('Train set evaluation for Ridge:\n_____________________________________')
print_evaluate(y_train, train_pred)
