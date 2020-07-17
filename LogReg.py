import math
import numpy as np
import torch
from torch.nn import Sigmoid, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from sklearn.base import BaseEstimator, ClassifierMixin

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class BaseLogRegClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, lr=0.01, num_iter=100000, batch_size=1, L=0, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.batch_size = batch_size # for SGD
        self.L = L # L2 coefficient
        self.verbose = verbose
        self.coef_ = None
    
    def fit(self, X, y):
        raise NotImplementedError()
    
    def predict_proba(X):
        raise NotImplementedError()
    
    def predict(X ,thershold):
        raise NotImplementedError()
        
    def score(X, y, threshold):
        raise NotImplementedError()

class NpLogRegClassifier(BaseLogRegClassifier):
    
    def fit(self, X, y): # X: (samples x features), y: (samples x 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        n_samples, n_features = X.shape 
        a = np.zeros((n_features, 1))
        for it in range(self.num_iter):
            loss = 0
            for bi in range(math.ceil(n_samples / self.batch_size)):
                X_i = X[self.batch_size * bi: min(self.batch_size * (bi + 1), n_samples), :]
                y_i = y[self.batch_size * bi: min(self.batch_size * (bi + 1), n_samples), :]
                pred = sigmoid(X_i @ a)
                loss += -y_i * np.log(pred) - (1 - y_i) * np.log(1 - pred) / self.batch_size
                grad = X_i.T @ (pred - y_i) / n_samples + 2 * a * self.L
                a -= self.lr * grad
            if it % 1000 == 0 and self.verbose:
                print(f"Loss on iteration {it}: {loss / n_samples}")
        self.coef_ = a # 0th weight corresponds to 1
    
    def predict_proba(self, X):
        if self.coef_ is None:
            raise Exception("Classifier not fitted")
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X @ self.coef_
    
    def predict(self, X ,threshold=0.5):
        if self.coef_ is None:
            raise Exception("Classifier not fitted")
        return (self.predict_proba(X) >= threshold).astype(np.int8)
    
    def score(self, X, y, threshold=0.5):
        if self.coef_ is None:
            raise Exception("Classifier not fitted")
        return (self.predict(X, threshold) == y).mean()
    
class CustomDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return {
                    'X': self.X[idx, :],
                    'y': self.y[idx, :]
               }
