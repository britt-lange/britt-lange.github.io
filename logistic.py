import torch
import numpy as np


class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        
        return self.w@X.T


    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        # Compute the scores
        scores = self.score(X)
        # Apply sigmoid function to convert to binary values
        return torch.sigmoid(scores).float() 
    
class LogisticRegression(LinearModel):
    
    def __init__(self):
        super().__init__()
        
    def loss(self, X, y):
        
        """
        Compute the binary cross-entropy loss.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
            where n is the number of data points and p is the
            number of features. This implementation always assumes
            that the final column of X is a constant column of 1s.
            y, torch.Tensor: the true labels. y.size() == (n,)

        RETURNS:
            loss, torch.Tensor: the binary cross-entropy (BCE) loss
        """
        
        s = self.score(X)
        s_ = s[:, None]
        # compute the predicted probabilities
        y_hat = torch.sigmoid(s)
        
        n = X.shape[0]
        
        # compute loss with binary cross entropy
        # loss = ((-1) * y * torch.log(y_hat) - (1 - y) * torch.log(1 - y_hat)).mean()
        loss = (-y@torch.log(y_hat).T - (1-y)@torch.log(1-y_hat).T) / n
        
        return loss
    
    def grad(self, X, y):
        
        """
        Compute the gradient of the binary cross-entropy loss.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
            where n is the number of data points and p is the
            number of features. This implementation always assumes
            that the final column of X is a constant column of 1s.
            y, torch.Tensor: the true labels. y.size() == (n,)

        RETURNS:
            grad, torch.Tensor: the gradient of the binary cross-entropy loss
        """
        
        # Compute scores and apply sigmoid
        s = self.predict(X)
        
        #convert size of y and scores matrix from (n,) to (n, 1)
        s_ = s[:, None]
        y_ = y[:, None]
        
        n = X.shape[0]

        # Compute gradient of the empirical risk
        grad = ((s - y)@X) / n
        
        return grad
    
class GradientDescentOptimizer():
    
    def __init__(self, model):
        self.model = model
        self.w_prev = None
    
    def step(self, X, y, alpha = 0.1, beta = 0.9):
        curr = self.model.w 
        if self.w_prev is None:
            self.model.w = self.model.w - (alpha * self.model.grad(X,y))
        else:
            self.model.w = self.model.w - (alpha * self.model.grad(X, y)) + (beta * (self.model.w - self.w_prev))
        self.w_prev = curr