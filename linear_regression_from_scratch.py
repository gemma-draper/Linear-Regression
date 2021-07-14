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

    def predict(self, X):
        """
        Calculate y_pred
        """
        return np.matmul(X, self.W) + self.b

    def _calc_MSE_loss(self, y_pred, y):
        """
        Calculates the mean squared error loss.
        """
        return np.mean((y_pred-y)**2)

    def _calc_gradients(self, X, y):
        """
        Calculate and return gradients of W and b.
        """
        # calculate gradient of weight and bias
        y_pred = self.predict(X)
        grad_b = 2 * np.mean(y_pred - y)
        grad_W = 2 * np.mean(np.matmul((y_pred - y), X))
        return grad_W, grad_b

        pass