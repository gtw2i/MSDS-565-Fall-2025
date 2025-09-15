import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#plt.style.use("dark_background")

def GenerateData(nPts,err,center):
    
    cov = err**2*np.eye(2)
    x1 = np.random.multivariate_normal(mean=[-center,-center],cov=cov,size=nPts//2)
    y1 = np.zeros(nPts//2)
    x2 = np.random.multivariate_normal(mean=[center,center],cov=cov,size=nPts//2)
    y2 = np.ones(nPts//2)
    
    X = np.concatenate((x1,x2),axis=0)
    y = np.concatenate((y1,y2),axis=0)
    
    return X, y
# end

#np.random.seed(0)

n_train = 50
n_test  = 50
center  = 1
err     = 0.7

# generate data
X_train, y_train = GenerateData(n_train,err,center)

# train model
nNeigh = 3
model = KNeighborsClassifier(n_neighbors=nNeigh)
model.fit(X_train,y_train)

# test model
X_test, y_test = GenerateData(n_test,err,center)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print("Training acc.:", acc_train)
print("Testing acc.: ", acc_test)

# plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,2*6))
axes = axes.flatten()

# plot training data
pltScale = 3
axes[0].scatter(X_train[:,0],X_train[:,1],c=y_train,s=40,cmap='RdBu')
axes[0].set_xlim(-pltScale,pltScale)
axes[0].set_ylim(-pltScale,pltScale)
axes[0].set_title("training data")

# plot regions
nGrid = 100
a1 = np.linspace(-pltScale,pltScale,nGrid)
a2 = np.linspace(-pltScale,pltScale,nGrid)

A1, A2 = np.meshgrid(a1,a2)
A1 = A1.flatten()
A2 = A2.flatten()
A = np.vstack((A1,A2)).T

B = model.predict(A)
B = B.reshape(nGrid,nGrid)
B = np.flipud(B)

extent = [-pltScale,pltScale,-pltScale,pltScale]

axes[0].imshow(B, interpolation='none', extent=extent,cmap='RdBu', alpha=0.5)

# plot test data
axes[1].scatter(X_test[:,0],X_test[:,1],c=y_test,s=40,cmap='RdBu')
axes[1].set_xlim(-pltScale,pltScale)
axes[1].set_ylim(-pltScale,pltScale)
axes[1].set_title("testing data")

axes[1].imshow(B, interpolation='none', extent=extent,cmap='RdBu', alpha=0.5)

plt.tight_layout()