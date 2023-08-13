import sys
sys.path.append("/home/jaxmao/JaxMao")

from jaxmao.Layers import Layer
from jaxmao.Activations import Activation

import jax.numpy as jnp
from jax import vmap
from jax import random, Array
from jax.random import PRNGKeyArray

from jaxmao.Layers import FC

class Module:    
    classes_that_have_params = (Layer, Activation)
    layers = list()
    params = list()
    
    def __init__(self, layers=None):
        pass # define layers here 
    
    def init_params(self, key : Array | PRNGKeyArray):
        self.layers = list()
        self.params = list()
        
        for layer in self.__dict__.values():
            if isinstance(layer, self.classes_that_have_params):
                key, subkey = random.split(key)
                layer.init_params(subkey)
                self.layers.append(layer)
                self.params.append(layer.params)
                
    def _forward(self, params, x):
        """
            Define the forward pass of one data point under this name.
        """
        pass
    
    def forward(self, params, x):
        return vmap(self._forward, in_axes=(None, 0))(params, x)
    
    def __call__(self, x):
        return self.forward(self.params, x)
        
    def params_forward(self, params, x):
        for i in range(len(params)):
            self.layers[i].params = params[i]
        return self.forward(x)

    
    def backward(self, x, y):
        pass 
        
if __name__ == '__main__':
    from jaxmao.Activations import ReLU, StableSoftmax
    from jax import random
    class MNIST_Classifier(Module):
        def __init__(self):
            self.fc1 = FC(784, 512)
            self.fc2 = FC(512, 256)
            self.fc3 = FC(256, 10)
            self.relu = ReLU()
            self.softmax = StableSoftmax()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.softmax(self.fc3(x))
            return x

    
    clf = MNIST_Classifier()
    clf.init_params(random.PRNGKey(42))
    # print(clf.params, type(clf.params))
    print(clf.layers)
