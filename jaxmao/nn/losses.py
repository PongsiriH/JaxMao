import jax.numpy as jnp
from jax import grad, jit

def mean_over_batch_size(negative_log_likelihood):
    sum_loss = jnp.sum(negative_log_likelihood)
    mean_loss = sum_loss / negative_log_likelihood.shape[0]
    return mean_loss

class Loss:
    def __init__(self, reduce_fn='mean_over_batch_size', eps=1e-12):
        super().__init__()
        self.eps = eps
        if reduce_fn is None:
            reduce_fn = lambda x: x
        elif reduce_fn == 'mean_over_batch_size':
            reduce_fn = mean_over_batch_size
        elif reduce_fn == 'mean':
            reduce_fn = jnp.mean
        elif reduce_fn == 'sum':
            reduce_fn = jnp.sum
        
        self.reduce_fn = reduce_fn

    def calculate_loss(self, y_pred, y_true):
        pass
 
    def __call__(self, y_pred, y_true, **kwargs):
        # assert y_pred.shape == y_true.shape, "Shape mismatch between y_pred and y_true"
        return self.calculate_loss(y_pred, y_true, **kwargs)

class MeanSquaredError(Loss):
    def calculate_loss(self, y_pred, y_true):
        return self.reduce_fn(jnp.square(y_true - y_pred))

class BinaryCrossEntropy(Loss):
    def calculate_loss(self, y_pred, y_true):
        y_pred_clipped = jnp.clip(y_pred, self.eps, 1 - self.eps)
        return -self.reduce_fn(y_true * jnp.log(y_pred_clipped + self.eps) + (1 - y_true) * jnp.log(1 - y_pred_clipped + self.eps))

class CategoricalCrossEntropy(Loss):
    def calculate_loss(self, y_pred, y_true):        
        y_pred_clipped = jnp.clip(y_pred, self.eps, 1 - self.eps)
        negative_log_likelihood = -jnp.log(y_pred_clipped) * y_true
        return self.reduce_fn(negative_log_likelihood)
