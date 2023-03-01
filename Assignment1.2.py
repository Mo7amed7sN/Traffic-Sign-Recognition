import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv('AssignmentData.csv')
X=data['duration']  * data['height']
Y=data['interval']


plt.scatter(X, Y)
plt.xlabel('duration * height', fontsize = 20)
plt.ylabel('interval', fontsize = 20)
plt.show()

model = linear_model.LinearRegression()
X=np.expand_dims(X, axis=1)
Y=np.expand_dims(Y, axis=1)
model.fit(X,Y)
prediction= model.predict(X)

plt.scatter(X, Y)
plt.xlabel('duration * height', fontsize = 20)
plt.ylabel('interval', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',model.coef_)
print('Intercept of linear regression model',model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

input_duration=float(input('Enter the duration : '))
input_hieght = int(input('Enter the Hieght : '))
x_test=np.array([input_duration * input_hieght])
x_test=np.expand_dims(x_test, axis=1)
y_test=model.predict(x_test)
print('the predicted interval is ' + str(int(y_test[0])))

