import numpy as np
from matplotlib import pyplot as plt

X = [[1500,1],[2000,2],[1650,1],[2200,3]]
X = np.array(X)
y = [210000,280000,260000,320000]
y = np.array(y)

data = np.zeros((4,3))
data[:,0:2] = X
data[:,2] = y
np.savetxt("SimpleData.txt", data, delimiter=',')

# plot the data
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,3))

axes[0].scatter(X[:,0],y,s=50)
axes[0].set_xlabel("area (ft^2)")
axes[0].set_ylabel("price ($)")

axes[1].scatter(X[:,1],y,s=50)
axes[1].set_xlabel("floors")
axes[1].set_ylabel("price ($)")

plt.tight_layout()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0],X[:,1],y,s=50)
ax.scatter(X[:,0],X[:,1],np.zeros_like(y),s=50)

# model the data
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
coef = model.coef_
b = model.intercept_

print(coef,b)

mins = np.min(X,axis=0)
maxs = np.max(X,axis=0)
x0 = np.linspace(mins[0],maxs[0],10)
x1 = np.linspace(mins[1],maxs[1],10)
X0, X1 = np.meshgrid(x0,x1)

# plot the model predictions
y2 = coef[0]*X0+coef[1]*X1+b
ax.plot_surface(X0,X1,y2)




