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
X_train, X_val, y_train, y_val = test_train_split(X_train, y_train)

# normalise the training data
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
# normalise the validation data
X_val = (X_val - X_mean) / X_std

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
    
    def fit(self, X, y, X_val, y_val, epochs=100, learning_rate=0.001, plot=True):
        """
        Fit linear regression model to X and y data.
        """
        train_loss = []
        val_loss = []

        batched_data = DataLoader(X, y) # batch the training data

        for epoch in range(epochs):
            train_loss_this_epoch = []
            val_loss_this_epoch = []

            for X_batch, y_batch in batched_data:
                y_hat, train_loss_this_epoch = self._get_loss(X_batch, y_batch, train_loss_this_epoch)
                val_loss_this_epoch = self._get_loss(X_val, y_val, val_loss_this_epoch, return_y_hat=False)
                grad_W, grad_b = self._calc_gradients(X_batch, y_batch, y_hat) # compute gradients 
                self._update_params(grad_W, grad_b, learning_rate) # update parameters
            
            
            val_loss.append(np.mean(val_loss_this_epoch))
            train_loss.append(np.mean(train_loss_this_epoch)) # append the mean loss for this epoch
                        
        if plot:

            plt.plot(train_loss, label="Training set")
            plt.plot(val_loss, label="Validation set")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.show()
                
    def predict(self, X):
        """
        Calculate y_hat
        """
        return np.matmul(X, self.W) + self.b
    
    def _get_loss(self, X, y, loss_this_epoch, return_y_hat=True):
        y_hat = self.predict(X) # make predictions
        loss = self._calc_MSE_loss(y_hat, y) # compute the loss
        loss_this_epoch.append(loss)
        if return_y_hat:
            return y_hat, loss_this_epoch
        return loss_this_epoch

    def _update_params(self, grad_W, grad_b, learning_rate):
        self.W -= learning_rate * grad_W # update the weight 
        self.b -= learning_rate * grad_b # update the bias 

    def _calc_MSE_loss(self, y_hat, y):
        """
        Calculates the mean squared error loss.
        """
        return np.mean((y_hat-y)**2)

    def _calc_gradients(self, X, y, y_hat):
        """
        Calculate and return gradients of W and b.
        """
        # calculate gradient of weight and bias
        grad_b = 2 * np.mean(y_hat - y)
        grad_W = 2 * np.mean(np.matmul((y_hat - y), X))
        return grad_W, grad_b




#%%
model = LinearRegression(n_features=X.shape[1])
model.fit(X_train, y_train, X_val, y_val)

# %%
