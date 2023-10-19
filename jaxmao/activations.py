import jax.numpy as jnp
from jaxmao.Layers import Layer

def relu(x):
    return jnp.maximum(0, x)

class Activation(Layer):
    def __init__(self):
        self.params = None
        self.num_params = None
        self.shapes = None
    def init_params(self, key):
        pass
    def _forward(self, params, x):
        return self.calculate(params, x)
    # def __call__(self, x):
    #     return self.calculate(self.params, x)
    
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