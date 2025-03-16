import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv('Samsung Ecosystem_1.csv')
X = dataset[['Year','DX','DS','MX']]
y = dataset['Revenue']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

clf = KNeighborsRegressor(11)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Prediction for test set: {}".format(y_pred))

reg_model_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(reg_model_diff)

print("The Mean Squared Error is:",mean_squared_error(y_test,y_pred))
print("The Root Mean Squared Error is:",root_mean_squared_error(y_test,y_pred))
print("The Mean Absolute Error is:",mean_absolute_error(y_test,y_pred))
print("R Squared Value is:",r2_score(y_test,y_pred))