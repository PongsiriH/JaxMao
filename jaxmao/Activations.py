import jax.numpy as jnp

class Activation:
    def __init__(self):
        self.params = None

class ReLU(Activation):        
    def __call__(self, params, x):
        return jnp.maximum(0, x)
    
class StableSoftmax(Activation):        
    def __call__(self, params, x):
        logits = x
        max_logits = jnp.max(logits, axis=self.axis, keepdims=True)
        shifted_logits = logits - max_logits
        exp_shifted_logits = jnp.exp(shifted_logits)
        softmax_probs = exp_shifted_logits / jnp.sum(exp_shifted_logits, axis=self.axis, keepdims=True)
        return softmax_probs