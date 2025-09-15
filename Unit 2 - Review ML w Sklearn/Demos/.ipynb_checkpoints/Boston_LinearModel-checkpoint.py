import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data  # Features (input variables)
y = boston.target  # Target variable (housing prices)
feature_names = boston.feature_names  # Feature names
description = boston.DESCR  # Dataset description

# print("Data shape:", X.shape)
# print("Target shape:", y.shape)
# print("Feature names:", feature_names)
# print("Dataset description:\n", description)

model = LinearRegression(fit_intercept=True)
model.fit(X, y)

y2 = model.predict(X)

r2 = r2_score(y,y2)
print(r2)
MSE = mean_squared_error(y,y2)
print(MSE)


