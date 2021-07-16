# %%
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

X, y = datasets.load_boston(return_X_y=True)

def test_train_split(X, y, test_size=0.2):
    """
    Split X and y data into traing set and test set.
    The default test set proportion is 0.2.
    Returns X_train, X_test, y_train, y_test
    """
    idx = 0
    length_of_X = len(X)
    y_test = []
    X_test = []
    
    while  idx < length_of_X*test_size:
        random_number_gen = np.random.randint(low=0, high=len(X))
        y_test.append(y[random_number_gen])
        X_test.append(X[random_number_gen])
        X = np.delete(X, random_number_gen, axis=0)
        y = np.delete(y, random_number_gen, axis=0)
        idx += 1
        return X, np.array(X_test), y, np.array(y_test)


# generate training, evaluation and test datasets.
X_train, X_test, y_train, y_test = test_train_split(X, y)
X_train, X_eval, y_train, y_eval = test_train_split(X_train, y_train)

# normalise the training data
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std

class DataLoader:
    """
    Data loader class: splits data into mini-batches.
    """
    def __init__(self, X, y):
        self.batch(X, y) # batch the data

    def batch(self, X, y, batch_size=16):
        """
        Shuffles and splits X and y data into batches.
        Default batch size is 16 examples.
        """
        X, y = self._shuffle(X, y) # shuffle the data
        self.batches = []
        idx = 0
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            self.batches.append(batch)
            idx += batch_size
 
    def _shuffle(self, X, y):
        """
        Private method. Shuffles the X and y arrays given.
        """
        X_and_y = np.c_[X, y]
        np.random.shuffle(X_and_y)
        return X_and_y[:, :-1], X_and_y[:, -1]
    
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

class LinearRegression:
    def __init__(self, n_features):
        # assign random params
        self.W = np.random.randn(n_features)
        self.b = np.random.randn()
    
    def fit(self, X, y, epochs=32, learning_rate=0.001):
        losses = []
        batched_data = DataLoader(X, y) # batch the data
        for epoch in range(epochs):
            loss_this_epoch = []
            for X_batch, y_batch in batched_data:
                y_pred = self.predict(X_batch) # make predictions
                loss = self._calc_MSE_loss(y_pred, y_batch) # compute the loss
                loss_this_epoch.append(loss)
                grad_W, grad_b = self._calc_gradients(X_batch, y_batch) # compute gradients 
                self.W -= learning_rate * grad_W # update the weight 
                self.b -= learning_rate * grad_b # update the bias 
            losses.append(np.mean(loss_this_epoch)) # append the mean loss for this epoch
        
        plt.plot(losses) # plot the mean loss against epoch number
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
model.fit(X_train,y_train)
# predictions = model.predict(X)
# %%
