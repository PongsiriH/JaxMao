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

class HeUniform(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype=jnp.float32):
        key, subkey = random.split(key)
        limit = jnp.sqrt(6/shape[0])
        return random.uniform(subkey, shape, dtype, -limit, limit)
    
class GlorotNormal(Initializer):
    def __call__(self,
                 key,
                 shape,
                 dtype=jnp.float32):
        key, subkey = random.split(key)
        if len(shape) == 1:
            n = (shape[0]+shape[0])
        else:
            n = (shape[0] + shape[1])
        return random.normal(subkey, shape, dtype) * jnp.sqrt(2/n)

class GlorotUniform(Initializer):
    def __call__(self,
                 key,
                 shape,
                 dtype=jnp.float32):
        key, subkey = random.split(key)
        if len(shape) == 1:
            n = (shape[0]+shape[0])
        else:
            n = (shape[0]+shape[1])
        limit = jnp.sqrt(6 / n)
        return random.uniform(subkey, shape, dtype, -limit, limit)
    

    