import jax.numpy as jnp
import jax

class Initializer:
    pass

class HeNormal(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype='float'):
        return jax.random.normal(key, shape, dtype) * jnp.sqrt(2/shape[0])

class Zeros(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype='float'):
        return jnp.zeros(shape, dtype)

class Consts(Initializer):
    def __init__(self, value):
        self.value = value
        
    def __call__(self, 
                 key, 
                 shape, 
                 dtype='float'):
        return jnp.full(shape, self.value, dtype) 