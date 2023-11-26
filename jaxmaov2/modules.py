from typing import Any
import jax.numpy as jnp
from jax import lax
import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_flatten_with_path
import jaxmaov2.initializers as init
from jaxmaov2.utils_struct import (PostInitialization,
                                   VariablesDict, Variable,
                                   )
import copy

import warnings

class PureContext:
    def __init__(self, module):
        # self.module = copy.deepcopy(module)
        self.module = copy.deepcopy(LightModule(module.__call__, module.submodules, module.taken_names_, module.params_, module.states_))
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        del self.module
        del self
        
class Bind:
    def __init__(self, module, params, states, trainable=False):
        self.module = copy.deepcopy(module)
        self.params = params
        self.states = states
        self.trainable = trainable

    def __enter__(self):
        self.module.set_trainable(self.trainable)
        _update(self.module, self.params, self.states)
        return self
    
    def __exit__(self, *args, **kwargs):
        del self.module
        del self.params
        del self.states
        del self

def _get_parameters(module):
    parameters = dict()
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True)
    for name in module.submodules:
        parameters[name] = _get_parameters(module.submodules[name])
    return parameters

def _get_states(module):
    states = dict()
    if len(module.submodules) == 0:
        return module.states_.get_value(as_dict=True)
    for name in module.submodules:
        states[name] = _get_states(module.submodules[name])
    return states

def _get_parameters_and_states(module):
    """more efficient  _get_parameters + _get_states"""
    parameters = dict()
    states = dict()
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True), module.states_.get_value(as_dict=True)
    for name in module.submodules:
        parameters[name], states[name] = _get_parameters_and_states(module.submodules[name])
    return parameters, states

def _get_states_and_regularizes(module):
    """more efficient _get_states + regularizes"""
    states = dict()
    regularizes = 0.0
    if len(module.submodules) == 0:
        return module.states_.get_value(as_dict=True), module.params_.get_reg_value()
    for name in module.submodules:
        states[name], reg = _get_states_and_regularizes(module.submodules[name])
        regularizes += reg
    return states, regularizes

def _get(module):
    """more efficient  _get_parameters + _get_states + regularizes"""
    parameters = dict()
    states = dict()
    regularizes = 0.0
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True), module.states_.get_value(as_dict=True), module.params_.get_reg_value()
    for name in module.submodules:
        parameters[name], states[name], regularizes = _get(module.submodules[name])
    return parameters, states, regularizes

def _update_parameters(module, params):
    if len(module.submodules) == 0:
        for key, param in params.items():
            module.params_.set_value(key, param)
        
    for name, submodule in module.submodules.items():
        _update_parameters(submodule, params[name])
        
def _update(module, params, states):
    if len(module.submodules) == 0:
        for name, param in params.items():
            module.params_.set_value(name, param)
            
        for name, state in states.items():
            module.states_.set_value(name, state)
        
    for name, submodule in module.submodules.items():
        _update(submodule, params[name], states[name])   
    
def _init(module, key):
    for name in module.params_():
        key, subkey = jax.random.split(key)
        shape, initializer = module.params_.get_meta(name)
        module.params_.set_value(name, initializer(subkey, shape, 'float32'))
        
    for name in module.states_():
        key, subkey = jax.random.split(key)
        shape, initializer = module.states_.get_meta(name)
        module.states_.set_value(name, initializer(subkey, shape, 'float32'))
        
    for name, submodule in module.submodules.items():
        key, subkey = jax.random.split(key)
        _init(submodule, subkey)

class LightModule:
    """make copy of a module with only neccesary parts for forward and backward pass"""
    def __init__(self, __call__, submodules, taken_names_, params_, states_):
        self.call = __call__
        if len(submodules) == 0:
            self.submodules = dict()
        else:
            self.submodules = {name: LightModule(submodules[name].__call__, submodules[name].submodules, submodules[name].taken_names_, submodules[name].params_, submodules[name].states_) for name in submodules.keys()}
        self.taken_names_ = taken_names_
        self.params_ = params_
        self.states_ = states_

    def __call__(self, inputs):
        return self.call(inputs)
    
    def get_reg_value(self, name):
        if name not in list(self.params_.keys()):
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_reg_value(name)
    
    def param(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_value(name)

    def state(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.states_.get_value(name)
    
class Module(metaclass=PostInitialization):
    is_collectable = True
    
    def __init__(self):
        self.submodules = dict()
        self.taken_names_ = []
        self.params_ = VariablesDict()
        self.states_ = VariablesDict()
        self.trainable = True
        
        self.num_submodules = 0
        self.num_params = 0
        self.num_states = 0
    
    def _post_init(self):
        for (attr_name, obj) in self.__dict__.items():
            if (hasattr(obj, 'is_collectable')
                and obj.is_collectable
                ):
                self.submodules[attr_name] = obj
                obj.name = attr_name
        self.num_submodules = len(self.submodules)
    
    def __call__(self, inputs):
        if self.trainable:
            return self.call(inputs)
        return jax.lax.stop_gradient(self.call(inputs))   
    
    def set_value(self, name, value):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        if name in list(self.params_.keys()):
            self.params_._dict[name]._value = value
        elif name in list(self.states_.keys()):
            self.states_._dict[name]._value = value

    def get_reg_value(self, name):
        if name not in list(self.params_.keys()):
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_reg_value(name)
            
    def param(self, name: str, shape=None, initializer=None, regularizer=None):
        """Manages the parameters of a model."""
        if shape is not None and initializer is not None:
            if name in self.taken_names_:
                raise ValueError(f"Name {name} is already taken.")
            
            self.taken_names_.append(name)
            self.params_[name] = Variable(shape, initializer, regularizer)
        elif name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")

        return self.params_.get_value(name)
    
    def state(self, name: str, shape=None, initializer=None):
        """Manages the states of a model."""
        if shape is not None and initializer is not None:
            if name in self.taken_names_:
                raise ValueError(f"Name {name} is already taken.")
            
            self.taken_names_.append(name)
            self.states_[name] = Variable(shape, initializer)
        elif name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")

        return self.states_.get_value(name)

    def init(self, key):
        with PureContext(self) as ctx:
            _init(ctx.module, key)
            _parameters, _states = _get_parameters_and_states(ctx.module)
        return _parameters, _states
        
    def structure(self):
        _structure = dict()
        if len(self.submodules) == 0:
            return self.params_, self.states_
        for name in self.submodules:
            _structure[name] = self.submodules[name].structure()
        return _structure

    def apply(self, inputs, params, states):
        with PureContext(self) as ctx:
            _update(ctx.module, params, states)
            out = ctx.module(inputs)
            states , reg = _get_states_and_regularizes(ctx.module)
        return out, states, reg
    
    def set_trainable(self, trainable):
        self.trainable = trainable
        for name in self.submodules:
            self.submodules[name].set_trainable(trainable)

class Dense(Module):
    def __init__(
        self,
        in_channles, out_channels,
        kernel_init=init.HeNormal(), 
        bias_init=init.Zeros(),
        kernel_reg=None,
        bias_reg=None,
        use_bias=True
    ):
        super().__init__()
        self.use_bias = use_bias
        self.out_channels = out_channels
        self.param('kernel', (in_channles, out_channels), kernel_init, kernel_reg)
        if self.use_bias:
            self.param('bias', (out_channels, ), bias_init, bias_reg)    
            
        self.kernel_reg = kernel_reg
        self.bias_reg = bias_reg
    
    def call(self, inputs):
        """Simple case: wx + b (if use_bias is True)"""
        outputs = lax.dot_general(inputs, self.param('kernel'), dimension_numbers=((len(inputs.shape)-1, 0), ((), ())))
        if self.use_bias:
            bias_shape = (1,) * (len(inputs.shape) - 1) + (self.out_channels,)
            outputs = lax.add(outputs, lax.broadcast_in_dim(self.param('bias'), bias_shape, (len(inputs.shape) - 1,)) )
        return outputs
    
class GeneralConv2d(Module):
    """Parent of Conv2d and DepthwiseConv2d"""
    def __init__(
        self,
        kernel_shape, 
        kernel_init,
        feature_group_count : int,
        strides,
        padding,
        dilation,
        use_bias, 
        bias_shape=None,
        bias_init=None,
        dtype='float32'
    ):
        super().__init__()
        self.strides = strides if isinstance(strides, tuple) else (strides, strides) 
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        padding = padding.upper()
        if not padding in ['SAME', 'SAME_LOWER', 'VALID']:
            warnings.warn(f"Unsupported padding type: {padding}. Using 'SAME' as default.")
            padding = 'SAME'
        self.padding = padding
        
        self.param('kernel', kernel_shape, kernel_init)
        if self.use_bias:
            self.param('bias', bias_shape, bias_init)
    
    def call(self, x):
        x = lax.conv_general_dilated(x, self.param('kernel'), 
                                     window_strides = self.strides, 
                                     padding = self.padding,
                                     lhs_dilation = None, 
                                     rhs_dilation = self.dilation,
                                     dimension_numbers = ('NHWC', 'HWIO', 'NHWC'),
                                     feature_group_count = self.feature_group_count)
        if self.use_bias:
            x = x + self.param('bias')
        return x

class Conv2d(GeneralConv2d):
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            strides=(1, 1),
            padding='SAME',
            dilation=(1, 1),
            use_bias=True, 
            kernel_init=init.HeNormal(),
            bias_init=init.Zeros(),
            dtype='float32'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_height, kernel_width = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
        kernnel_shape = (kernel_height, kernel_width, in_channels, out_channels)
        self.use_bias = use_bias
        bias_shape = (out_channels,) if self.use_bias else None

        super().__init__(kernnel_shape, kernel_init,
                         feature_group_count=1,
                         strides=strides,
                         padding=padding,
                         dilation=dilation,
                         use_bias=use_bias,
                         bias_shape=bias_shape,
                         bias_init=bias_init,
                         dtype=dtype)

class BatchNormalization(Module):
    def __init__(
        self,
        num_features,
        momentum,
        axis_mean,
        running_mean_init,
        running_var_init,
        offset_init,
        scale_init,
        eps=1e-6
    ):
        super().__init__()
        self.momentum = momentum
        self.axis_mean = axis_mean
        self.eps = eps
        self.state('running_mean', (num_features, ), running_mean_init)
        self.state('running_var', (num_features, ), running_var_init)
        self.param('scale', (num_features, ), scale_init)
        self.param('offset', (num_features, ), offset_init)

    def call(self, inputs):
        if self.trainable:
            batch_mean = jnp.mean(inputs, axis=self.axis_mean)
            batch_var = jnp.var(inputs, axis=self.axis_mean)
            self.set_value('running_mean', self.momentum * self.state('running_mean') + (1 - self.momentum) * batch_mean)
            self.set_value('running_var', self.momentum * self.state('running_var') + (1 - self.momentum) * batch_var)
            normalized_x = (inputs - batch_mean) / jnp.sqrt(batch_var + self.eps)
        else:
            normalized_x = (inputs - self.state('running_mean')) / jnp.sqrt(self.state('running_var') + self.eps)
        return normalized_x * self.param('scale') + self.param('offset')

class BatchNorm1d(BatchNormalization):
    def __init__(
        self,
        num_features,
        momentum=0.99,
        running_mean_init=init.Consts(0),
        running_var_init=init.Consts(1),
        offset_init=init.Consts(0),
        scale_init=init.Consts(1),
        eps=1e-6
    ):
        super().__init__(num_features, momentum, 0, running_mean_init, running_var_init, offset_init, scale_init, eps)
        
class BatchNorm2d(BatchNormalization):
    def __init__(
        self,
        num_features,
        momentum=0.99,
        running_mean_init=init.Consts(0),
        running_var_init=init.Consts(1),
        offset_init=init.Consts(0),
        scale_init=init.Consts(1),
        eps=1e-6
    ):
        super().__init__(num_features, momentum, (0, 1, 2), running_mean_init, running_var_init, offset_init, scale_init, eps)

class Dropout(Module):
    def __init__(self, drop_rate=0.5, key=None):
        super().__init__()
        if key is None:
            warnings.warn('Dropout key is not provided. Proceed with default key. Be careful with randomness.')
            key = jax.random.key(42)
        self.state('drop_rate', None, None)
        self.state('key', None, None)
        self.set_value('drop_rate', drop_rate)
        self.set_value('key', key)
    
    def call(self, inputs):
        keep_rate = 1 - self.state('drop_rate')
        key, subkey = jax.random.split(self.state('key'))
        mask = jax.random.bernoulli(subkey, keep_rate, inputs.shape)
        self.set_value('key', key)
        return inputs * mask / keep_rate
        
class Pooling2d(Module):
    def __init__(
        self, kernel_size, strides, padding, init_value
    ):
        super().__init__()
        self.kernel_size = kernel_size + (1, )
        self.strides = strides + (1, )
        self.init_value = init_value
        self.padding = padding.upper() if isinstance(padding, str) else padding
        self.reducing_fn = None

    def _pool_forward(self, inputs):        
        return lax.reduce_window(
            inputs,
            init_value=self.init_value,
            computation=self.reducing_fn,
            window_dimensions=self.kernel_size,
            window_strides=self.strides,
            padding=self.padding
        )
        
class MaxPooling2d(Pooling2d):
    def __init__(
        self, kernel_size=(2, 2), strides=(2, 2), padding='SAME', 
        init_value=jnp.finfo('float32').min
    ):
        super().__init__(kernel_size, strides, padding=padding, init_value=init_value)
        self.reducing_fn = lax.max
    
    def __call__(self, inputs):
        return jax.vmap(self._pool_forward, in_axes=0)(inputs)      

class AveragePooling2d(Pooling2d):
    def __init__(
        self, kernel_size=(2, 2), strides=(2, 2), padding='SAME', 
        init_value=0.0
    ):
        super().__init__(kernel_size, strides, padding=padding, init_value=init_value)
        self.reducing_fn = lax.add
        self.kernel_prod = jnp.prod(jnp.array(kernel_size), dtype='float32')

    def call(self, inputs):
        summed_feature = jax.vmap(super()._pool_forward, in_axes=0)(inputs)
        return lax.div(summed_feature, self.kernel_prod)

class GlobalMaxPooling2d(Module):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return jnp.max(inputs, axis=(1, 2))

class GlobalAveragePooling2d(Module):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return jnp.mean(inputs, axis=(1, 2), keepdims=False)

if __name__ == '__main__':
    pass
    # print('\ndense.__call__:\n', jax.make_jaxpr(dense.__call__)(sample_inputs))
    # print('\ndense.call:\n', jax.make_jaxpr(dense.call)(sample_inputs))