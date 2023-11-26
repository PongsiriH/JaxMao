import jax.numpy as jnp

class Regularizer:
    def __call__(self, weights):
        return self.call(weights)

class L1(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def call(self, weights):
        """
        Return the L1 regularization term.
        L1 regularization term: alpha * sum(abs(weights))
        """
        return self.alpha * jnp.sum(jnp.abs(weights))

class L2(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def call(self, weights):
        """
        Return the L2 regularization term.
        L2 regularization term: alpha * sum(square(weights))
        """
        return self.alpha * jnp.sum(jnp.square(weights))

class L1L2(Regularizer):
    def __init__(self, alpha1, alpha2):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    
    def call(self, weights):
        return self.alpha1 * jnp.sum(jnp.abs(weights)) + self.alpha2 * jnp.sum(jnp.square(weights))