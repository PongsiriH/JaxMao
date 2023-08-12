import jaxmao.Layers as Layers

import jax.numpy as jnp
from jax import random, Array
from jax.random import PRNGKey

from jaxmao.Layers import FC

class Module:    
    def __init__(self, layers=None):
        self.fc = FC(5,5)
        pass # define layers here 
    
    def init_params(self, key : Array | PRNGKey):
        for layer in self.__dict__.values():
            print('hello')
            key, subkey = random.split(key)
            layer.init_params(subkey)
    
    def forward(self):
        pass # define the model here
    
    def backward(self):
        pass 
        
if __name__ == '__main__':
    m = Module()
    print(m)
    m.init_params(random.PRNGKey(5))
