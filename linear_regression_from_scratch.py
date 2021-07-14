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
    def __init__(self, n_features):
        # assign random params
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()
        pass
    
    def fit(self, X, y, epochs=32):
        losses = []
        for epoch in range(epochs):
            learning_rate = 0.001
            y_pred = self.predict(X) # make predictions
            loss = self._calc_MSE_loss(y_pred, y) # compute the loss
            losses.append(loss)

            grad_W, grad_b = self._calc_gradients(X, y) # compute gradients 
            self.W -= learning_rate * grad_W # update the weight 
            self.b -= learning_rate * grad_b # update the bias 
        
        plt.plot(losses) # plot the loss for each epoch
        plt.show()
        

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


model = LinearRegression(n_features=X.shape[1])
model.fit(X,y)
predictions = model.predict(X)
# %%
