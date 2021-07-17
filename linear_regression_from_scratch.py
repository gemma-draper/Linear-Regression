# %%
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

X, y = datasets.load_boston(return_X_y=True)

def test_train_split(X, y, test_size=0.2):
    """
    Split X and y data into traing set and test set.
    The default test set proportion is 0.2.
    Inputs: 
        X: 2D feature array.
        y: 1D label array.
    Outputs: 
        X_train, X_test, y_train, y_test
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

# standardise the training data
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
# standardise the validation data
X_val = (X_val - X_mean) / X_std

class DataLoader:
    """
    Data loader class: suffles and splits data into mini-batches.
    """
    def __init__(self, X, y):
        """
        Shuffle and batch the data.
        Batched data is stored in self.batches list. 
        Inputs: 
            X: 2D feature array.
            y: 1D label array.
        """
        self._batch(X, y)

    def _batch(self, X, y, batch_size=16):
        """
        Shuffles and splits X and y data into batches.
        Default batch size is 16 examples.
        Batched data is stored in self.batches list. 
        Inputs: 
            X: 2D feature array.
            y: 1D label array.
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
        Shuffles feature and label arrays.
        Inputs: 
            X: 2D feature array.
            y: 1D label array.
        Outputs:
            X, y

        """
        X_and_y = np.c_[X, y]
        np.random.shuffle(X_and_y)
        return X_and_y[:, :-1], X_and_y[:, -1]
    
    def __len__(self):
        """
        Defines behaviour for calling len() of a class instance.
        """
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Defines behaviour for indexing a class instance.
        """
        return self.batches[idx]

class LinearRegression:
    def __init__(self, n_features):
        """
        Constructs linear regression model. 
        Model is initialised with random weights and bias.
        """
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
                y_hat, train_loss_this_epoch = self._compute_loss(X_batch, y_batch, train_loss_this_epoch)
                val_loss_this_epoch = self._compute_loss(X_val, y_val, val_loss_this_epoch, return_y_hat=False)
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
        Calculate label predictions, y_hat, for features X.
        Input:
            X: 2D feature array.
        Output:
            y_hat: 1D array of label predictions.
        """
        return np.matmul(X, self.W) + self.b
    
    def _compute_loss(self, X, y, loss_this_epoch, return_y_hat=True):
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
