import jax.numpy as jnp
import jax

class Initializer:
    pass

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

class ValueInitializer(Initializer):
    def __init__(self, value):
        self.value = value        
    def __call__(self, 
                 key, 
                 shape,
                 dtype=None):
        return jnp.array(self.value)
    
class HeNormal(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype='float'):
        return jax.random.normal(key, shape, dtype) * jnp.sqrt(2/shape[0])
    
class HeUniform(Initializer):
    def __call__(self, 
                 key, 
                 shape, 
                 dtype='float32'):
        key, subkey = jax.random.split(key)
        limit = jnp.sqrt(6/shape[0])
        return jax.random.uniform(subkey, shape, dtype, -limit, limit)
    
class GlorotUniform(Initializer):
    def __call__(self,
                 key,
                 shape,
                 dtype='float32'):
        key, subkey = jax.random.split(key)
        if len(shape) == 1: # 
            fan_avg  = (shape[0]+shape[0])
        elif len(shape) == 2: # dense layer
            fan_avg  = (shape[0]+shape[1])
        elif len(shape) == 4: # convolution
            n_in = shape[0] * shape[1] * shape[2]
            n_out = shape[0] * shape[1] * shape[3]
            fan_avg  = n_in + n_out
        else:
            raise ValueError("Unrecognized tensor shape.")
        limit = jnp.sqrt(6 / fan_avg)

        return jax.random.uniform(subkey, shape, dtype, -limit, limit)