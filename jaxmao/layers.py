import jax
from .initializers import *
import jax.numpy as jnp
from jax import vmap, lax
from jax import random
import numpy as np


class Layer:
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype
        self.params = None
        self.shapes = None
        self.initializers = None
        self.num_params = None
        self.state = None
    
    def init_params(self, key):
        if self.shapes:
            self.params = dict()
            for layer in self.shapes.keys():
                key, subkey = random.split(key)
                self.params[layer] = self.initializers[layer](subkey, self.shapes[layer], dtype=self.dtype)
              
    def __call__(self, params, x):
        return self.forward(params, x, self.state)
        
class Dense(Layer):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        activation=None,
        weights_initializer=HeNormal(),
        bias_initializer=HeNormal(),
        use_bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        if not activation:
            activation = lambda x: x
        self.activation = activation
        
        self.shapes = {'weights' : (in_channels, out_channels)}
        self.initializers = {'weights' : weights_initializer}
        if self.use_bias:
            self.shapes['biases'] = (out_channels, )
            self.initializers['biases'] = bias_initializer
        
    def forward(self, params, x, state):
        z = jnp.dot(x, params['weights']) + params['biases']
        return self.activation(z), None
         
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
                
            self.forward = jax.jit(self.forward)
        def forward(self, params, x):
            # ('NCHW', 'OIHW', 'NCHW')
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=(self.strides, self.strides),
                                            padding=self.padding,
                                            dimension_numbers=('NCHW', 'OIHW', 'NHWC')) 
            if self.use_bias:
                x = lax.add(x, params['biases'][None, None, None, :]) # (batch_size, width, height, out_channels)
            return x

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, params, x):
        return x.reshape(x.shape[0], -1)

"""
    Batch Normalizations
"""
def gamma_initializer(key, shape, dtype):
    return jnp.ones(shape, dtype)        
def beta_initializer(key, shape, dtype):
    return jnp.zeros(shape, dtype)

class BatchNorma1D(Layer):
    def __init__(
        self,
        num_features,
        momentum = 0.5,
        dtype=jnp.float32,
        eps=1e-5
        ):
        super().__init__()
        self.moving_mean = 0
        self.moving_var = 1
        self.momentum = momentum
        self.eps = eps
        
        self.shapes = {
            'gamma' : (1, num_features),
            'beta'  : (num_features, ),
        }
        self.initializers = {
            'gamma' : gamma_initializer,
            'beta' : beta_initializer
        }
        self.dtype = dtype
    
    def forward(self, params, x, training=False):
        x, (moving_mean, moving_var) = x
        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            moving_mean = self.momentum * moving_mean + (1 - self.momentum) * batch_mean
            moving_var = self.momentum * moving_var + (1 - self.momentum) * batch_var
            return jnp.multiply(
                        (x - batch_mean) / jnp.sqrt(batch_var + self.eps), params['gamma']
                    ) + params['beta'], (moving_mean, moving_var)
        return jnp.multiply(
                (x - moving_mean) / jnp.sqrt(moving_var + self.eps), params['gamma']
            ) + params['beta']
    
    def _update_running_statistics(self, running_statistics):
        moving_mean, moving_var = running_statistics
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        
    # def __call__(self, x, training=False):
    #     results = self.forward(self.params, x, (self.moving_mean, self.moving_var), training=training)
    #     if len(results) == 2 and type(results[1]) == tuple and self._update_running_statistics:
    #         self._update_running_statistics(results[1])
    #         results = results[0]
    #     return results





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
            

    # def _forward(self, params, x):
    #     h = jnp.zeros((self.out_channels,), dtype=self.dtype)
    #     for current_x in x:
    #         h = lax.add(lax.dot(h, params['weights_h']), lax.dot(current_x, params['weights_x']))
    #         if self.use_bias:
    #             h = lax.add(h, params['biases_h'])
    #         h = lax.tanh(h)
    #     return h
    def _forward(self, params, x):
        # python for-loop equivalent.
        # for current_x in x:
        #     h = lax.add(lax.dot(h, params['weights_h']), lax.dot(current_x, params['weights_x']))
        #     if self.use_bias:
        #         h = lax.add(h, params['biases_h'])
        #     h = lax.tanh(h)
        # return h
        h = jnp.zeros((self.out_channels,), dtype=self.dtype)
        def rnn_step(h, current_x):
            h_new = lax.add(lax.dot(h, params['weights_h']), lax.dot(current_x, params['weights_x']))
            if self.use_bias:
                h_new = lax.add(h_new, params['biases_h'])
            h_new = lax.tanh(h_new)
            return h_new, None

        final_h, _ = lax.scan(rnn_step, h, x)
        return final_h

class LSTM(Layer):
    def __init__(
            self, 
            num_features,
            out_channels,
            weights_initializer=GlorotNormal(),
            bias_initializer=GlorotNormal(),
            use_bias=True,
            dtype=jnp.float32
        ):
        super().__init__()
        self.num_features = num_features
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.dtype = dtype

        self.shapes = {
            'weights_f': (num_features+out_channels, out_channels),
            'weights_i': (num_features+out_channels, out_channels),
            'weights_c': (num_features+out_channels, out_channels),
            'weights_o': (num_features+out_channels, out_channels),
        }
        self.initializers = {
            'weights_f': weights_initializer,
            'weights_i': weights_initializer,
            'weights_c': weights_initializer,
            'weights_o': weights_initializer,
        }
        if use_bias:
            self.shapes['biases_f'] = (out_channels,)
            self.shapes['biases_i'] = (out_channels,)
            self.shapes['biases_c'] = (out_channels,)
            self.shapes['biases_o'] = (out_channels,)
            self.initializers['biases_f'] = bias_initializer
            self.initializers['biases_i'] = bias_initializer
            self.initializers['biases_c'] = bias_initializer
            self.initializers['biases_o'] = bias_initializer

    def _forward(self, params, x):
        h = jnp.zeros((self.out_channels,), dtype=self.dtype)
        c = jnp.zeros((self.out_channels,), dtype=self.dtype)
        def lstm_step(carry, current_x):
            previous_h, previous_c = carry
            combined_input = jnp.concatenate([current_x, previous_h], axis=0)
            current_i = jax.nn.sigmoid(lax.dot(combined_input, params['weights_i']))
            current_f = jax.nn.sigmoid(lax.dot(combined_input, params['weights_f']))
            current_o = jax.nn.sigmoid(lax.dot(combined_input, params['weights_o']))
            current_cbar = jnp.tanh(lax.dot(combined_input, params['weights_c']))
            current_c = jnp.multiply(current_f, previous_c) + jnp.multiply(current_i, current_cbar)
            current_h = jnp.multiply(current_o, jnp.tanh(current_c))
            return (current_h, current_c), None
        (h_final, c_final), _ = lax.scan(lstm_step, (h, c), x)
        return h_final


        
class SimpleEmbedding(Layer):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        weights_initializer=HeNormal(),
        use_bias=True,
        dtype=jnp.float32
    ):
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim

        self.shapes = {'weights' : (vocab_size, embedding_dim)}
        self.initializers = {'weights' : weights_initializer}
        self.dtype = dtype
        
    def _forward(self, params, x):
        return jnp.take(params['weights'], x, axis=0)

# class SimpleEmbedding(Layer):
#     def __init__(
#             self,
#             vocab_size,
#             embedding_dim,
#             weights_initializer=HeNormal(),
#             use_bias=True,
#             dtype=jnp.float32
#     ):
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#
#         self.shapes = {'weights': (vocab_size, embedding_dim)}
#         self.initializers = {'weights': weights_initializer}
#         self.dtype = dtype
#
#     def _forward(self, params, x):
#         return jnp.take(params['weights'], x, axis=0)
