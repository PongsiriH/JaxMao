from jax import random
import jax.numpy as jnp


def ones_initializer(key, shape, dtype):
    return jnp.ones(shape, dtype)   
     
def zeros_initializer(key, shape, dtype):
    return jnp.zeros(shape, dtype)

def zeros_plus_initializer(key, shape, dtype):
    return jnp.full(shape, 0.01, dtype=dtype)


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
        elif len(shape) == 4:
            n_in = shape[0] * shape[1] * shape[2]
            n_out = shape[0] * shape[1] * shape[3]
            n = n_in + n_out
        else:
            n = (shape[0] + shape[1])
        print('{:<20} {:<10}'.format(str(shape), n))
        return random.normal(subkey, shape, dtype) * jnp.sqrt(2/n)
    
class GlorotUniform(Initializer):
    def __call__(self,
                 key,
                 shape,
                 dtype=jnp.float32):
        key, subkey = random.split(key)
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

        return random.uniform(subkey, shape, dtype, -limit, limit)

    