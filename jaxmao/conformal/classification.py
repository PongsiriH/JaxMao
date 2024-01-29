import jax
import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import label_binarize

cp_clf_key = jax.random.key(10242)

class ConformalNaive:
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        """
        y: label encoded.
        """
        self.n_training_samples = len(X)
        est_prob_prediction = self.estimator.predict_proba(X)
        non_conformity_score = jax.nn.softmax(est_prob_prediction, axis=1)
        self.conformity_score = 1 - non_conformity_score[jnp.arange(self.n_training_samples), y.astype(int)] 
        
    def predict(self, X, alpha):
        q_level = np.ceil((self.n_training_samples+1)*(1-alpha)) / self.n_training_samples
        qhat = np.quantile(self.conformity_score, q_level, method='higher')
        
        est_prob_prediction = self.estimator.predict_proba(X)
        prediction_sets = jax.nn.softmax(est_prob_prediction, axis=1) >= (1-qhat)
        return est_prob_prediction, prediction_sets
    
class ConformalLAC:
    def __init__(self, estimator=None):
        self.estimator = estimator
        self.conformity_score = None
        
    def fit(self, X, y):
        """
        y: label encoded.
        """
        self.n_training_samples = len(X)
        est_prob_prediction = self.estimator.predict_proba(X)
        self.conformity_score = 1 - est_prob_prediction[jnp.arange(self.n_training_samples), y.astype(int)] # only take where the target is.
    
    def predict(self, X, alpha):
        q_level = jnp.ceil((self.n_training_samples + 1) * (1 - alpha)) / self.n_training_samples
        quantile = jnp.quantile(self.conformity_score, q_level, method='higher')
                
        est_prob_prediction = self.estimator.predict_proba(X)
        pred_sets = est_prob_prediction >= (1 - quantile)
        return est_prob_prediction, pred_sets

class ConformalAPS:
    def __init__(self, estimator=None):
        self.estimator = estimator
        self.conformity_score = None
        
    def fit(self, X, y, rand=True):
        self.n_training_samples  = len(X)
        
        y_enc = label_binarize(y)
        estimated_score = self.estimator.predict_proba(X)
        index_sorted = jnp.argsort(estimated_score, axis=1)[:, ::-1]
        estimated_score_sorted = jnp.take_along_axis(estimated_score, index_sorted, axis=1)
        estimated_score_cumsum = jnp.cumsum(estimated_score_sorted, axis=1)

        y_enc_sorted = jnp.take_along_axis(y_enc, index_sorted, axis=1)
        cutoff = jnp.argmax(y_enc_sorted, axis=1)
        self.conformity_score = jnp.take_along_axis(estimated_score_cumsum, cutoff.reshape(-1, 1), axis=1)
        if rand:
            global cp_clf_key
            cp_clf_key, subkey = jax.random.split(cp_clf_key)
            y_proba_true = np.take_along_axis(estimated_score, y_enc.reshape(-1, 1), axis=1)
            self.conformity_score = self.conformity_score - jax.random.uniform(subkey, len(y_proba_true)) * y_proba_true
        

    def predict(self, X, alpha):
        quantile = jnp.quantile(self.conformity_score, np.ceil((self.n_training_samples + 1) * (1 - alpha)) / self.n_training_samples, method="higher")
        
        estimated_score = self.estimator.predict_proba(X)
        index_sorted = jnp.argsort(estimated_score, axis=1)[:, ::-1]
        estimated_score_sorted = jnp.take_along_axis(estimated_score, index_sorted, axis=1)
        estimated_score_cumsum = jnp.cumsum(estimated_score_sorted, axis=1)
        prediction_sets = jnp.take_along_axis(estimated_score_cumsum <= quantile, estimated_score_sorted.argsort(axis=1), axis=1)
        return estimated_score, prediction_sets
    
class ConformalRAPS:
    pass