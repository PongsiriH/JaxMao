import sys
sys.path.append("/home/jaxmao/JaxMao")

from jaxmao.Layers import Layer
from jaxmao.Activations import Activation

import jax.numpy as jnp
from jax import vmap
from jax import random, Array
from jax.random import PRNGKeyArray

class Module:    
    classes_that_have_params = (Layer, Activation)
    layers = None
    params = None
    
    def __init__(self, layers=None):
        """
            Define all layers you will use here.
        """
        pass
    
    def init_params(self, key : Array | PRNGKeyArray):
        self.layers = list()
        self.params = list()
        
        for layer in self.__dict__.values():
            if isinstance(layer, self.classes_that_have_params):
                key, subkey = random.split(key)
                layer.init_params(subkey)
                self.layers.append(layer)
                self.params.append(layer.params)
                
    def __call__(self, x):
        """
            Define the behavior of forward pass of one datapoint 
            under this __call__ function.          
        """
        pass
                  
    def forward(self, params, x):
        """
            forward function update `params of each layer` to the `provided params`.
            Then predict using __call__.
            
            This function is particularly useful for training. We can take grad()
            with respect to the `provided params`.
        """
        for i in range(len(params)):
            self.layers[i].params = params[i]
        return self.__call__(x)
    
"""
    Example:

    class MNIST_Classifier(Module):
        def __init__(self):
            self.conv1 = Conv2D(1, 32, 3, 2) 
            self.flatten = Flatten()
            self.fc1 = FC(32*14*14, 32)
            self.fc2 = FC(32, 10)
            self.relu = ReLU()
            self.softmax = StableSoftmax()
            
        def __call__(self, x):
            x = self.relu(self.conv1(x))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.softmax(self.fc2(x))
            return x
"""
