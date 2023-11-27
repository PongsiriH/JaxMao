from typing import Any
import jax.numpy as jnp
from jax import lax
import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_flatten_with_path
import jaxmao.initializers as init
from jaxmao.utils_struct import (PostInitialization, VariablesDict, Variable, LightModule)
from jaxmao.module_utils import (
                                 _get_parameters, _get_states, _get_parameters_and_states, _get_states_and_regularizes, _get,
                                _update_parameters, _update,
                                _init, _init_zero,
                                module_id
                                )
import pickle, os, copy, warnings

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
        self.params_ = params
        self.states_ = states
        self.trainable = trainable

    def __enter__(self):
        self.module.set_trainable(self.trainable)
        _update(self.module, self.params_, self.states_)
        return self
    
    def __exit__(self, *args, **kwargs):
        del self.module
        del self.params_
        del self.states_
        del self

class Save:
    def __init__(self, path):
        self.path = path
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        del self.path
        del self
    
    def save(self, item, file_name):
        path = os.path.join(self.path, file_name)
        if os.path.exists(path):
            raise FileExistsError(f"File '{file_name}' already exists.")
        with open(path, 'wb') as f:
            pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(self, file_name):
        path = os.path.join(self.path, file_name)
        with open(path, 'rb') as f:
            loaded_item = pickle.load(f)
        return loaded_item
    
class Summary:
    """How should I save each layer?"""
    def __init__(self, module):
        self.module = copy.deepcopy(module)
        _init_zero(self.module)
        self.params_, self.states_ = _get_parameters_and_states(self.module)
        
    def __enter__(self):
        self.module.set_trainable(False)
        _update(self.module, self.params_, self.states_)
        return self
    
    def summary(self, input_shape):
        x = jnp.zeros(input_shape)
        self.module(x)
        
    def __exit__(self, *args, **kwargs):
        del self.module
        del self.params_
        del self.states_

class Module(metaclass=PostInitialization):
    is_collectable = True
    
    def __init__(self):
        self.submodules = dict()
        self.taken_names_ = []
        self.params_ = VariablesDict()
        self.states_ = VariablesDict()
        self.trainable = True
        
        self._id = module_id()
        self.num_submodules = 0
        self.num_params = 0
        self.num_states = 0
    
    def _post_init(self):
        for (attr_name, obj) in self.__dict__.items():
            if (hasattr(obj, 'is_collectable') and obj.is_collectable):
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


class Sequential(Module):
    def __init__(self, modules: list=None):
        super().__init__()
        if modules is None:
            modules = []
        else:
            # if not all(isinstance(module, Module) for module in modules):
            #     raise ValueError("All elements in 'modules' must be instances of 'Module'")
            modules = list(modules)
        self.submodules = {}
        [self.add(module) for module in modules]        
    
    # def _post_init(self):
    #     super()._post_init()
    #     print('post init self.modules', self.modules)
    #     self.submodules = {name: self.modules[name] for name in self.modules}
    
    def call(self, x):
        for name in self.submodules:
            x = self.submodules[name](x)
        return x

    def add(self, module):
        self.submodules[module._id] = module
    

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

"""Poolings: actually, just use flax. it make more sense to take a functional approach."""
def max_pool_2d(x, window_shape, strides):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1,)+window_shape+(1,), (1,)+strides+(1,), 'VALID')

class Pooling2d(Module):
    def __init__(self, kernel_size, strides, padding):
        super().__init__()
        self.kernel_size = (1, ) + kernel_size + (1, ) if isinstance(kernel_size, tuple) else (1, kernel_size, kernel_size, 1)
        self.strides = (1, ) + strides + (1, ) if isinstance(strides, tuple) else (1, strides, strides, 1)
        self.padding = padding.upper() if isinstance(padding, str) else padding

class MaxPooling2d(Pooling2d):
    def __init__(self, kernel_size=(2, 2), strides=(2, 2), padding='SAME'):
        super().__init__(kernel_size, strides, padding=padding)
    
    def call(self, inputs):
        return lax.reduce_window(inputs, -jnp.inf, lax.max, self.kernel_size, self.strides, self.padding)

class AveragePooling2d(Pooling2d):
    def __init__(self, kernel_size=(2, 2), strides=(2, 2), padding='SAME'):
        super().__init__(kernel_size, strides, padding=padding)
        self.kernel_prod = jnp.prod(jnp.array(kernel_size), dtype='float32')

    def call(self, inputs):
        summed_feature = lax.reduce_window(inputs, 0.0, lax.add, (1,)+self.window_shape+(1,), (1,)+self.strides+(1,), self.padding)
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
        return jnp.mean(inputs, axis=(1, 2))


class LeakyReLU(Module):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def call(self, inputs):
        return jax.nn.leaky_relu(inputs, self.alpha)

if __name__ == '__main__':
    pass
    # print('\ndense.__call__:\n', jax.make_jaxpr(dense.__call__)(sample_inputs))
    # print('\ndense.call:\n', jax.make_jaxpr(dense.call)(sample_inputs))