# %%
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
X, y = datasets.load_boston(return_X_y=True)

print(X.shape)
print(y.shape)

# normalise X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X  = (X - X_mean)/X_std

class LinearRegression:
    def __init__(self, n_features=13):
        # assign random params
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()
        pass
    
    def fit(self):
        # for epoch in epochs:
            # make predictions
            # compute the loss
            # compute gradient of the loss hyperplane
            # update the weight and bias
        pass

    def predict(self):
        # calculate y_hat
        pass

    def _calc_MSE_loss(self):
        # calculate loss 
        pass

    def _calc_gradients(self):
        # calculate gradient of weight and bias
        pass