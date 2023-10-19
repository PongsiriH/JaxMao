import jax.numpy as jnp
from jax import grad, jit

class Loss:
    @jit
    def __call__(self, y_pred, y_true):
        return self.calculate_loss(y_pred, y_true)

class MeanSquaredError(Loss):
    def calculate_loss(self, y_pred, y_true):
        return jnp.inner(y_true-y_pred, y_true-y_pred).mean()

class CategoricalCrossEntropy(Loss):
    def calculate_loss(self, y_pred, y_true):
        return -jnp.mean(
            jnp.sum(y_true*jnp.log(y_pred))
            )