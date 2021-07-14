# %%
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

X, y = datasets.load_boston(return_X_y=True)

# print(X.shape)
# print(y.shape)

# normalise X
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X  = (X - X_mean)/X_std

class DataLoader:
    """
    Data loader class: splits data into mini-batches.
    """
    def __init__(self, X, y):
        self.batch(X, y) # batch the data

    def batch(self, X, y, batch_size=16):
        X, y = self._shuffle(X, y)
        self.batches = []
        idx = 0
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            self.batches.append(batch)
            idx += batch_size

    def __len__(self):
        return len(self.batches)
    
    def _shuffle(self, X, y):
        X_and_y = np.c_[X, y]
        np.random.shuffle(X_and_y)
        return X_and_y[:, :-1], X_and_y[:, -1]

    def __getitem__(self, idx):
        return self.batches[idx]

class LinearRegression:
    def __init__(self, n_features):
        # assign random params
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()
        pass
    
    def fit(self, X, y, epochs=32):
        losses = []
        learning_rate = 0.001
        batched_data = DataLoader(X, y)

        for epoch in range(epochs):
            loss_this_epoch = []
            for X_batch, y_batch in batched_data:
                y_pred = self.predict(X_batch) # make predictions
                loss = self._calc_MSE_loss(y_pred, y_batch) # compute the loss
                loss_this_epoch.append(loss)
                grad_W, grad_b = self._calc_gradients(X_batch, y_batch) # compute gradients 
                self.W -= learning_rate * grad_W # update the weight 
                self.b -= learning_rate * grad_b # update the bias 
            losses.append(np.mean(loss_this_epoch))
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



#%%
model = LinearRegression(n_features=X.shape[1])
model.fit(X,y)
predictions = model.predict(X)
# %%
