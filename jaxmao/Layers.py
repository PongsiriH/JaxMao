from jaxmao.Initializers import HeNormal
import numpy as np

import jax.numpy as jnp
from jax import vmap, lax
from jax import random

class Layer:
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype
        self.params = None
        self.shapes = None
        self.initializers = None
        self.num_params = None
        
    def forward(self, params, x):
        return vmap(self._forward, in_axes=(None, 0))(params, x)
    
    def __call__(self, x):
        return self.forward(self.params, x)
    
    def init_params(self, key):
        if self.shapes:
            self.params = dict()
            for layer in self.shapes.keys():
                key, subkey = random.split(key)
                self.params[layer] = self.initializers[layer](subkey, self.shapes[layer], dtype=self.dtype)
            
    def count_params(self):
        self.num_params = 0
        if self.shapes:
            for layer in self.shapes.keys():
                self.num_params = self.num_params + np.prod(self.shapes[layer])
        return self.num_params

class FC(Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 weights_initializer=HeNormal(),
                 bias_initializer=HeNormal(),
                 use_bias=True,
                 dtype=jnp.float32
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.dtype = dtype
        
        self.shapes = {'weights' : (in_channels, out_channels)}
        self.initializers = {'weights' : weights_initializer}
        if self.use_bias:
            self.shapes['biases'] = (out_channels, )
            self.initializers['biases'] = bias_initializer
            
            
    def _forward(self, params, x):
        x = x.astype(self.dtype)
        x = lax.dot(x, params['weights'])
        if self.use_bias:
            x = lax.add(x, params['biases'])
        return x
         
class Conv2D(Layer):
        def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    strides=1, 
                    use_bias=True, 
                    padding='SAME',
                    weights_initializer=HeNormal(),
                    bias_initializer=HeNormal(),
                    dtype=jnp.float32
            ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.strides = strides
            self.use_bias = use_bias
            self.padding = padding
            self.dtype = dtype
            
            self.shapes = dict()
            self.initializers = dict()
            if isinstance(kernel_size, int):
                self.shapes['weights'] = (out_channels, in_channels, kernel_size, kernel_size)
            elif isinstance(kernel_size, tuple):
                self.shapes['weights'] = (out_channels, in_channels, kernel_size[0], kernel_size[1])
            self.initializers['weights'] = weights_initializer

            if use_bias:
                self.shapes['biases'] = (out_channels, )
                self.initializers['biases'] = bias_initializer

        def forward(self, params, x):
            # ('NCHW', 'OIHW', 'NCHW')
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=(self.strides, self.strides),
                                            padding=self.padding) 
            if self.use_bias:
                x = lax.add(x, params['biases'][None, :, None, None]) # (batch_size, out_channels, width, height)
            return x

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def _forward(self, params, x):
        return x.ravel()

"""
    Recurrent Neural Network
"""
class SimpleRNN(Layer):
    def __init__(
            self, 
            num_features,
            out_channels,
            weights_initializer=HeNormal(),
            bias_initializer=HeNormal(),
            use_bias=True,
            dtype=jnp.float32
        ):
        super().__init__()
        self.num_features  = num_features
        self.out_channels  = out_channels
        self.use_bias      = use_bias
        self.dtype         = dtype
        
        self.shapes = {
            'weights_x' : (num_features, out_channels),
            'weights_h' : (out_channels, out_channels),
        }
        self.initializers = {
            'weights_x' : weights_initializer,
            'weights_h' : weights_initializer,
        }
        if use_bias:
            self.shapes['biases_h'] = (out_channels,)
            self.initializers['biases_h'] = bias_initializer
            

    def _forward(self, params, x):
        h  = jnp.zeros((self.out_channels,), dtype=self.dtype)        
        for current_x in x:
            h = lax.add(lax.dot(params['weights_h'].T, h), lax.dot(params['weights_x'].T, current_x))
            if self.use_bias:
                h = lax.add(h, params['biases_h'])
            h = lax.tanh(h)
        return h