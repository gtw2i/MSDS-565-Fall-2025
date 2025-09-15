import numpy as np
from matplotlib import pyplot as plt

X = [1500,2000,1650,2200]
X = np.array(X)
X = X[:,np.newaxis]
y = [210000,280000,260000,320000]
y = np.array(y)

# plot the data
fig = plt.figure(figsize=(6,6))
plt.scatter(X[:,0],y,s=50)
plt.xlabel("area (ft^2)")
plt.ylabel("price ($)")

# model the data
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
coef = model.coef_
b = model.intercept_

print(coef,b)

# plot the model predictions
mins = np.min(X,axis=0)
maxs = np.max(X,axis=0)
x0 = np.linspace(mins[0],maxs[0],10)

y2 = coef[0]*x0+b
plt.plot(x0,y2,'r')




