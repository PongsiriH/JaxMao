from jax import random
import jax.numpy as jnp

class Initializer:
    pass

class HeNormal(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype=jnp.float32):
        key, subkey = random.split(key)
        return random.normal(subkey, shape, dtype) * jnp.sqrt(2/shape[0])