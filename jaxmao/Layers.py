from jaxmao.Initializers import HeNormal

import jax.numpy as jnp
from jax import vmap, lax
from jax import random

class Layer:
    def __init__(self):
        self.params = None
    
    def forward(self, params, x):
        return vmap(self._forward, in_axes=(None, 0))(params, x)
    
    def __call__(self, params, x):
        return self.forward(params, x)
    
class FC(Layer):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 weights_initializer=HeNormal(),
                 bias_initializer=HeNormal(),
                 use_bias=True,
                 dtype=jnp.float32
        ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.dtype=dtype
        
        self.shape = dict()
        self.shape['weights'] = (out_channels, in_channels)
        if self.use_bias:
            self.shape['bias'] = (out_channels, )
        self.params = dict()
   
    def init_params(self, key):
        key, w_key, b_key = random.split(key, 3)
        self.params['weights'] = self.weights_initializer(w_key, self.shape['weights'], self.dtype)
        if self.use_bias:
            self.params['bias'] = self.bias_initializer(b_key, self.shape['bias'], self.dtype)
            
    def _forward(self, params, x):
        x = x.astype(self.dtype)
        return jnp.add(
            lax.dot(x, params['weights']), params['bias']
        )
        
class Conv2D(Layer):
        def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    strides=1, 
                    use_bias=True, 
                    padding='valid',
                    weights_initializer=HeNormal(),
                    bias_initializer=HeNormal(),
                    dtype=jnp.float32
            ):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.strides = strides
            self.use_bias = use_bias
            self.padding = padding
            self.weights_initializer = weights_initializer 
            self.bias_initializer = bias_initializer
            self.dtype = dtype
            
            self.shape = dict()
            if isinstance(kernel_size, int):
                self.shape['weights'] = (out_channels, in_channels, kernel_size, kernel_size)
            elif isinstance(kernel_size, tuple):
                self.shape['weights'] = (out_channels, in_channels, kernel_size[0], kernel_size[1])
            if use_bias:
                self.shape['bias'] = (out_channels, )
            self.params = dict()

            
        def init_params(self, key):
            key, w_key, b_key = random.split(key, 3)
            self.params['weights'] = self.weights_initializer(w_key, self.shape['weights'], self.dtype)
            self.params['bias'] = self.bias_initializer(b_key, self.shape['bias'], self.dtype)
        
        def forward(self, params, x):
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=(self.stride, self.stride),
                                            padding="SAME")
            if self.use_bias:
                x = lax.add(x, params['bias'][None, :, None, None])
            return x

class Flatten(Layer):
    def _forward(self, params, x):
        return x.ravel()

if __name__ == '__main__':
    fc = FC(5, 5)
    print(fc)
    # print(for )