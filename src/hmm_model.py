#Completed? Y

from hmmlearn import hmm
import numpy as np

class HMMModel:
    def __init__(self, n_components):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)
        
    def train(self, X):
        self.model.fit(X)
        
    def predict(self, X):
        return self.model.predict(X)
