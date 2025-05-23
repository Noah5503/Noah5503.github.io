import torch

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
            s, torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return X @ self.w
    
class LogisticRegression(LinearModel):

    def loss(self, X, y):
        """
        Compute the logistic loss.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. 
            y, torch.Tensor: Shape (n,), with binary labels (0 or 1).

        RETURNS:
            loss, scalar tensor (average logistic loss)
        """
        score = self.score(X)
        y_hat = torch.sigmoid(score)
        return torch.mean(-y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. 
            y, torch.Tensor: Shape (n,), with binary labels (0 or 1).

        RETURNS:
            grad, torch.Tensor: shape (p,)
        """
        score = self.score(X)
        y_hat = torch.sigmoid(score)
        n = X.size(0)
        grad = torch.zeros(X.size(1))

        for i in range(n):
            grad += (y_hat[i] - y[i]) * X[i]
        grad /= n # take avg
        
        return grad
    
class GradientDescentOptimizer():

    def __init__(self, model):
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        """
        Perform a step of the gradient descent optimizer with momentum.

        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
            where n is the number of data points and p is the
            number of features.
            y, torch.Tensor: Shape (n,), with binary labels (0 or 1).
            alpha, float: the learning rate. Scales the extent to which
            the loss affects the weight vector during the update.
            beta, float: the momentum term. Scales the amount of 
            momentum used in updating w.

        RETURNS:
            loss, scalar tensor: the loss at a given step 
            (used to track progress over time)
        """
        old_w = self.model.w 
        loss = self.model.loss(X, y)
        if self.prev_w == None:
            self.prev_w = torch.rand((X.size()[1]))
        self.model.w = self.model.w - ((alpha*loss) + beta*(self.model.w - self.prev_w))
        self.prev_w = old_w
        return loss
