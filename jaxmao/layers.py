import warnings

import jax
from .initializers import HeNormal
import jax.numpy as jnp
from jax import vmap, lax
from jax import random
import numpy as np

class Layer:
    def __init__(self, dtype=jnp.float32):
        self.dtype = dtype
        self.params = dict()
        self.shapes = dict()
        self.initializers = None
        self.num_params = 0
        self.num_states = 0
        self.state = dict()
        self.layers = None
        self.summary = '{:<20} {:<20} {:<20} {:<20}\n'.format('layer', 'output shape', '#\'s params', '#\'s states')
        
    def init_params(self, key=None):
        if key is None:
            key = random.PRNGKey(0)
        if self.shapes:
            self.params = dict()
            for layer in self.shapes.keys():
                key, subkey = random.split(key)
                self.params[layer] = self.initializers[layer](subkey, self.shapes[layer], dtype=self.dtype)
        if self.layers:
            for name in self.layers:
                key, subkey = random.split(key)
                self.layers[name].init_params(subkey)
                self.params[name] = self.layers[name].params                
    
    def __call__(self, params, x):
        return self.forward(params, x, self.state)

    def apply(self, params, x, name, state=None):
        if name in self.layers:
            layer = self.layers[name]
            if isinstance(params, dict):
                if name in params:
                    x, layer_state = layer(params[name], x)
                    state[name] = layer_state
                else:
                    x, layer_state = layer(params, x)
                    state = layer_state
            self.summary += '{:<20} {:<20} {:<20} {:<20}\n'.format(name, str(x.shape), layer.count_params(), layer.num_states)
            self.num_params += layer.num_params
        return x, state

    def count_params(self):
        self.num_params = 0
        if self.shapes:
            for layer in self.shapes.keys():
                self.num_params = self.num_params + np.prod(self.shapes[layer])
        return self.num_params

    def set_training_mode(self):
        self.state['training'] = True
    
    def set_inference_mode(self):
        self.state['training'] = False
        
class Dense(Layer):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        activation='relu',
        weights_initializer=HeNormal(),
        bias_initializer=HeNormal(),
        use_bias=True,
    ):
        super().__init__()
        if activation is None or activation in ['linear']:
            activation = Linear()
        elif activation == 'relu':
            activation = ReLU()
        self.activation = activation

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        
        self.shapes = {'weights' : (in_channels, out_channels)}
        self.initializers = {'weights' : weights_initializer}
        if self.use_bias:
            self.shapes['biases'] = (out_channels, )
            self.initializers['biases'] = bias_initializer
        
    def forward(self, params, x, state=None):
        z = jnp.dot(x, params['weights'])
        if self.use_bias:
            z = z + params['biases']
        return self.activation(z), None
    
class Conv2D(Layer):
        def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    strides=(1, 1),
                    activation='relu',
                    padding='SAME',
                    dilation=(1, 1),
                    use_bias=True, 
                    weights_initializer=HeNormal(),
                    bias_initializer=HeNormal(),
                    dtype=jnp.float32
            ):
            super().__init__()
            self.forward = jax.jit(self.forward)
            
            if not padding in ['SAME', 'SAME_LOWER', 'VALID']:
                warnings.warn(f"Unsupported padding type: {padding}. Using 'SAME' as default.")
                self.padding = 'SAME'
            
            if activation in [None, 'linear']:
                activation = Linear()
            elif activation == 'relu':
                activation = ReLU()
            self.activation = activation
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.strides = strides if isinstance(strides, tuple) else (strides, strides)
            self.use_bias = use_bias
            self.padding = padding
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            self.dtype = dtype
            
            self.shapes = dict()
            self.initializers = dict()
            if isinstance(kernel_size, int):
                self.shapes['weights'] = (kernel_size, kernel_size, in_channels, out_channels)
            elif isinstance(kernel_size, tuple):
                self.shapes['weights'] = (kernel_size[0], kernel_size[1], in_channels, out_channels)
            self.initializers['weights'] = weights_initializer

            if use_bias:
                self.shapes['biases'] = (out_channels, )
                self.initializers['biases'] = bias_initializer
                
        def forward(self, params, x, state=None):
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=self.strides,
                                            padding=self.padding,
                                            lhs_dilation=None,
                                            rhs_dilation=self.dilation,
                                            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
                                            ) 
            if self.use_bias:
                x = lax.add(x, params['biases'][None, None, None, :]) # (batch_size, width, height, out_channels)
            return self.activation(x), None

class DepthwiseConv2D(Layer):
        def __init__(self, 
                    in_channels, 
                    depth_multiplier=1,
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    activation='relu',
                    padding='SAME',
                    dilation=(1, 1),
                    use_bias=True, 
                    weights_initializer=HeNormal(),
                    bias_initializer=HeNormal(),
                    dtype=jnp.float32
            ):
            super().__init__()
            self.forward = jax.jit(self.forward)
            if not padding in ['SAME', 'SAME_LOWER', 'VALID']:
                warnings.warn(f"Unsupported padding type: {padding}. Using 'SAME' as default.")
                self.padding = 'SAME'
            
            if activation in [None, 'linear']:
                activation = Linear()
            elif activation == 'relu':
                activation = ReLU()
            self.activation = activation
            
            self.in_channels = in_channels
            self.depth_multiplier = depth_multiplier
            self.out_channels = in_channels * depth_multiplier
            self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            self.strides = strides if isinstance(strides, tuple) else (strides, strides)
            self.use_bias = use_bias
            self.padding = padding
            self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            self.dtype = dtype
            
            self.shapes = dict()
            self.initializers = dict()
            if isinstance(kernel_size, int):
                self.shapes['weights'] = (kernel_size, kernel_size, self.depth_multiplier, in_channels)
            elif isinstance(kernel_size, tuple):
                self.shapes['weights'] = (kernel_size[0], kernel_size[1], self.depth_multiplier, in_channels)
            self.initializers['weights'] = weights_initializer

            if use_bias:
                self.shapes['biases'] = (self.out_channels, )
                self.initializers['biases'] = bias_initializer
        
        def forward(self, params, x, state=None):
            x = lax.conv_general_dilated(x, params['weights'], 
                                            window_strides=self.strides,
                                            padding=self.padding,
                                            lhs_dilation=None,
                                            rhs_dilation=self.dilation,
                                            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                            feature_group_count=self.out_channels
                                            ) 
            if self.use_bias:
                # Broadcasting the biases across the batch and spatial dimensions.
                # have out_channels biases.
                # (batch_size, width, height, out_channels)
                x = lax.add(x, params['biases'][None, None, None, :]) 
            return self.activation(x), None

class DepthwiseSeparableConv2D(Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            depth_multiplier=1,
            activation='relu',
            padding='SAME',
            strides=(1, 1),
            dilation=(1, 1),
            use_bias=True,
            weights_initializer=HeNormal(),
            bias_initializer=HeNormal(),
            dtype=jnp.float32
                 ):
        super().__init__()
        self.layers = {
            'depthwise' : DepthwiseConv2D(
                                in_channels, depth_multiplier, 
                                kernel_size=(3, 3), strides=strides, 
                                activation=activation, padding=padding,
                                dilation=dilation, use_bias=use_bias,
                                weights_initializer=weights_initializer,
                                bias_initializer=bias_initializer,
                                dtype=dtype
                                          ),
            'pointwise' : Conv2D(
                            in_channels, out_channels, 
                            kernel_size=(1, 1), strides=(1, 1),
                            activation=activation,
                            dilation=dilation, use_bias=use_bias,
                            weights_initializer=weights_initializer,
                            bias_initializer=bias_initializer,
                            dtype=dtype
                                 )
        }
    
    def forward(self, params, x, state=None):
        x, state = self.apply(params, x, 'depthwise', state)
        x, state = self.apply(params, x, 'pointwise', state)
        return x, state

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, params, x, state=None):
        return x.reshape(x.shape[0], -1), None

"""
    Batch Normalizations
"""
def gamma_initializer(key, shape, dtype):
    return jnp.ones(shape, dtype)        
def beta_initializer(key, shape, dtype):
    return jnp.zeros(shape, dtype)
        
class BatchNorm(Layer):
    def __init__(
        self,
        num_features,
        momentum = 0.5,
        axis_mean = 0,
        eps=1e-5
        ):
        super().__init__()
        self.axis_mean = axis_mean
        self.eps = eps
        self.num_states = num_features + num_features
        
        self.state = {
            'running_mean' : jnp.zeros((num_features,)),
            'running_var' : jnp.ones((num_features,)),
            'momentum' : momentum,
            'training' : True
        }
        
        self.shapes = {
            'gamma' : (num_features, ),
            'beta'  : (num_features, ),
        }
        self.initializers = {
            'gamma' : gamma_initializer,
            'beta' : beta_initializer
        }
        
    def forward(self, params, x, state):
        def training_mode(_):
            batch_mean = jnp.mean(x, axis=self.axis_mean)
            batch_var = jnp.var(x, axis=self.axis_mean)
            new_running_mean = state['momentum'] * state['running_mean'] + (1 - state['momentum']) * batch_mean
            new_running_var = state['momentum'] * state['running_var'] + (1 - state['momentum']) * batch_var
            return new_running_mean, new_running_var, batch_mean, batch_var
        def inference_mode(_):
            return state['running_mean'], state['running_var'], state['running_mean'], state['running_var']

        new_running_mean, new_running_var, batch_mean, batch_var = lax.cond(
                                                                        state['training'], 
                                                                        None, training_mode, 
                                                                        None, inference_mode
                                                                            )
        normalized_x = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
        scaled_x = normalized_x * params['gamma'] + params['beta']
        
        new_state = {
            'running_mean': new_running_mean,
            'running_var': new_running_var,
            'momentum' : state['momentum'],
            'training' : state['training']
        }
        
        return scaled_x, new_state

class BatchNorm2D(BatchNorm):
    def __init__(
        self,
        num_features,
        momentum = 0.5,
        eps=1e-5
        ):
        super().__init__(num_features=num_features, momentum=momentum, axis_mean=(0, 1, 2), eps=eps)
        
"""
    Activation layers
"""
"""
    Activation functions
"""
class Activation(Layer):
    def __init__(self):
        super().__init__()
        self.params = None
        self.num_params = None
        self.shapes = None
        
    def init_params(self, key):
        pass
    
    def forward(self, params, x, state=None):
        return self.calculate(params, x), state
    
    def __call__(self, x, state=None):
        return self.calculate(params=None, x=x)

class Linear(Activation):
    def __init__(self):
        super().__init__()
        
    def calculate(self, params, x):
        return x
    
class ReLU(Activation):
    def __init__(self):
        super().__init__()
    def calculate(self, params, x):
        return jnp.maximum(0, x)

class StableSoftmax(Activation):    
    def __init__(self):
        super().__init__()
        
    def calculate(self, params, x, axis=-1):
        logits = x
        max_logits = jnp.max(logits, axis=axis, keepdims=True)
        shifted_logits = logits - max_logits
        exp_shifted_logits = jnp.exp(shifted_logits)
        softmax_probs = exp_shifted_logits / jnp.sum(exp_shifted_logits, axis=axis, keepdims=True)
        return softmax_probs