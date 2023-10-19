import jax
from .initializers import HeN
import jax.numpy as jnp
from jax import vmap, lax
from jax import random
import numpy as np


import jax.numpy as jnp
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

def gamma_initializer(key, shape, dtype):
    return jnp.ones(shape, dtype)        
def beta_initializer(key, shape, dtype):
    return jnp.zeros(shape, dtype)

class BatchNorm(Layer):
    def __init__(
        self,
        num_features,
        momentum = 0.5,
        eps=1e-5
        ):
        super().__init__()
        self.eps = eps
        
        self.state = {
            'running_mean' : jnp.zeros((num_features,)),
            'running_var' : jnp.ones((num_features,)),
            'momentum' : momentum,
            'training' : True
        }
        
        self.shapes = {
            'gamma' : (1, num_features),
            'beta'  : (num_features, ),
        }
        self.initializers = {
            'gamma' : gamma_initializer,
            'beta' : beta_initializer
        }
        
    def forward(self, params, x, state):
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        
        new_running_mean = state['running_mean']
        new_running_var = state['running_var']
        if state['training']:
            new_running_mean = state['momentum'] * state['running_mean'] + (1 - state['momentum']) * batch_mean
            new_running_var = state['momentum'] * state['running_var'] + (1 - state['momentum']) * batch_var
        
        normalized_x = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
        scaled_x = normalized_x * params['gamma'] + params['beta']
        
        new_state = {
            'running_mean': new_running_mean,
            'running_var': new_running_var,
            'momentum' : state['momentum'],
            'training' : state['training']
        }
        
        return scaled_x, new_state


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
