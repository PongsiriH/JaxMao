from jaxmao.Initializers import HeNormal

import jax.numpy as jnp
from jax import vmap, lax
from jax import random

class Layer:
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype

        self.params = dict()
        self.shapes = dict()
        self.initializers = dict()
        
    def forward(self, params, x):
        return vmap(self._forward, in_axes=(None, 0))(params, x)
    
    def __call__(self, x):
        return self.forward(self.params, x)
    
    def init_params(self, key):
        for layer in self.shapes.keys():
            key, subkey = random.split(key)
            self.params[layer] = self.initializers[layer](subkey, self.shapes[layer], dtype=self.dtype)
            
    
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
        
        self.shapes = dict()
        self.shapes['weights'] = (in_channels, out_channels)
        if self.use_bias:
            self.shapes['biases'] = (out_channels, )
            
        self.initializers = {
            'weights' : weights_initializer,
            'biases' : bias_initializer
        }
   
    # def init_params(self, key):
    #     self.params = dict()
        
    #     key, w_key, b_key = random.split(key, 3)
    #     self.params['weights'] = self.initializers['weights'](w_key, self.shapes['weights'], self.dtype)
    #     if self.use_bias:
    #         self.params['biases'] = self.initializers['biases'](b_key, self.shapes['biases'], self.dtype)
            
    def _forward(self, params, x):
        x = x.astype(self.dtype)
        return jnp.add(
            lax.dot(x, params['weights']), params['biases']
        )
        
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
            if isinstance(kernel_size, int):
                self.shapes['weights'] = (out_channels, in_channels, kernel_size, kernel_size)
            elif isinstance(kernel_size, tuple):
                self.shapes['weights'] = (out_channels, in_channels, kernel_size[0], kernel_size[1])
            if use_bias:
                self.shapes['biases'] = (out_channels, )
                
            self.initializers = {
                'weights' : weights_initializer,
                'biases' : bias_initializer
            }
            
        # def init_params(self, key):
        #     self.params = dict()
        #     key, w_key, b_key = random.split(key, 3)
        #     self.params['weights'] = self.initializers['weights'](w_key, self.shapes['weights'], self.dtype)
        #     self.params['biases'] = self.initializers['biases'](b_key, self.shapes['biases'], self.dtype)
        
        def forward(self, params, x):
            # ('NCHW', 'OIHW', 'NCHW')
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=(self.strides, self.strides),
                                            padding=self.padding) 
            if self.use_bias:
                x = lax.add(x, params['biases'][None, :, None, None])
            return x

class Flatten(Layer):
    def _forward(self, params, x):
        return x.ravel()

class SimpleRNN(Layer):
    def __init__(
            self, 
            input_channel, 
            output_channel,
            number_layer,
            use_bias = True,
            dtype=jnp.float32
        ):
        super().__init__()
        self.input_channel  = input_channel
        self.output_channel = output_channel
        self.number_layer   = number_layer
        self.use_bias       = use_bias
        self.dtype = dtype
        
        self.shape = {
            'weights_x' : (input_channel, output_channel),
            'weights_h' : (output_channel, output_channel),
        }
        if use_bias:
            self.shape['biases_h'] = (output_channel, 1)
        self.params = None
        
    def init_params(self, key):
        Wx_key, Wh_key, bh_key, key = random.split(key, 4)
        self.params = {
            'weights_x' : random.normal(Wx_key, self.shape['weights_x']),
            'weights_h' : random.normal(Wh_key, self.shape['weights_h'])
        }
        if self.use_bias:
            self.params['biases_h'] = random.normal(bh_key, self.shape['biases_h'])

    def _forward(self, x):
        h  = jnp.zeros(self.output_channel, dtype=self.dtype)

        h = self.params['weights_h'] * h + self.params['weights_x'] * x
        if self.use_bias:
            h = h + self.params['biases_h']
        h = lax.tanh(h)
        
