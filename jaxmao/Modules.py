import sys
sys.path.append("/home/jaxmao/JaxMao")

from jaxmao.Layers import Layer
from jaxmao.Activations import Activation

import jax.numpy as jnp
from jax import vmap
from jax import random, Array
from jax.random import PRNGKeyArray

class InitializeLayersModule(type):
    """
        Help Module class initialize layers without having to explicitly declared.
    """
    def __call__(cls, *args, **kwargs):
        instance = super(InitializeLayersModule, cls).__call__(*args, **kwargs)
        instance.post_initialization()
        return instance

class Module(metaclass=InitializeLayersModule):    
    classes_that_have_params = (Layer, Activation)
    layers = None
    params = None
    
    def __init__(self, layers=None):
        """
            Define all layers you will use here.
        """
        self.num_layers = None
        self.num_params = None
        
    def post_initialization(self):
        self.init_layers()
        
    def init_layers(self):
        self.layers = list()
        self.params = list()
        for layer in self.__dict__.values():
            if isinstance(layer, self.classes_that_have_params):
                self.layers.append(layer)
        self.num_layers = len(self.layers)
        
        
    def init_params(self, key : Array | PRNGKeyArray):            
        for layer in self.layers:
            key, subkey = random.split(key)
            layer.init_params(subkey)
            self.params.append(layer.params)
        self.num_layers = len(self.layers)
        
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
            
            len(params) = number_of_layers in the module
        """
        for i in range(len(params)):
            self.layers[i].params = params[i]
        return self.__call__(x)
    
    def count_params(self):
        self.num_params = 0
        for layer in self.layers:
            self.num_params = self.num_params + layer.count_params()
        return self.num_params


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
