import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SGtSNEpi(BaseEstimator, TransformerMixin):
    """
    Stochastic Gradient t-SNE with Preservation of Interactions

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors
    learning_rate : float, default=200.0
        The learning rate for t-SNE
    n_iter : int, default=1000
        Maximum number of iterations for the optimization
    """

    def __init__(
        self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit_transform(self, X, y=None):
        """
        Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # Implementation will go here
        pass

    def fit(self, X, y=None):
        """
        Fit X into an embedded space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X)
        return self
