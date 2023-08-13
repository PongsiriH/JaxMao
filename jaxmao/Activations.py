import jax.numpy as jnp

class Activation:
    def __init__(self):
        self.params = None
    
    def init_params(self, key):
        pass
    
    def __call__(self, x):
        return self.calculate(self.params, x)
    
class ReLU(Activation):        
    def calculate(self, params, x):
        return jnp.maximum(0, x)
    
class StableSoftmax(Activation):        
    def calculate(self, params, x, axis=-1):
        logits = x
        max_logits = jnp.max(logits, axis=axis, keepdims=True)
        shifted_logits = logits - max_logits
        exp_shifted_logits = jnp.exp(shifted_logits)
        softmax_probs = exp_shifted_logits / jnp.sum(exp_shifted_logits, axis=axis, keepdims=True)
        return softmax_probs