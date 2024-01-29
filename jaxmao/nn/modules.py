from typing import Any

import numpy as np
import jax.numpy as jnp
from jax import lax
import jax
from jax.tree_util import tree_flatten, tree_unflatten, tree_flatten_with_path
from param import output
import torch
import jaxmao.nn.initializers as init
from jaxmao.nn.utils_struct import (PostInitialization, VariablesDict, Variable, LightModule)
from jaxmao.nn.module_utils import (
                                 _get_parameters, _get_states, _get_parameters_and_states, _get_states_and_regularizes, _get,
                                _update_parameters, _update,
                                _init, _init_zero,
                                UniqueID
                                )
import pickle, os, copy, warnings
from functools import partial
MAX_UINT32 =  4_294_967_295
# global id
# ____id = {
#     'Sequential',
#     'Dense',
#     'Conv2d',
#     'BatchNorm1d',
#     'BatchNorm2d',
#     'Dropout',
#     'MaxPooling2d',
#     'AveragePooling2d',
#     'GlobalMaxPooling2d',
#     'GlobalAveragePooling2d'
# }
class PureContext:
    def __init__(self, module):
        self.module = copy.deepcopy(LightModule(module.__call__, module.submodules, module.taken_names_, module.params_, module.states_))
        # self.module = copy.deepcopy(module)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        del self.module
        del self

@partial(jax.jit, static_argnums=(0, 4))
def apply(module, inputs, params, states, method_fn):
    _update(module, params, states)
    out = method_fn(inputs)
    states , reg = _get_states_and_regularizes(module)
    return out, states, reg
    
class Train:
    def __init__(self, module, params, states):
        self.module = copy.deepcopy(LightModule(module.__call__, module.submodules, module.taken_names_, module.params_, module.states_))
        _update(self.module, params, states)
        
    def __enter__(self):
        return self
    
    def apply(self, inputs, params, states, method_fn=None):
        if method_fn is None:
            method_fn = self.module.__call__
        return apply(self.module, inputs, params, states, method_fn)

    def __exit__(self, *args, **kwargs):
        del self.module.params_
        del self.module.states_
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
    
    def predict(self, inputs, batch_size=None):
        if batch_size is None:
            return self.module(inputs)

        num_batches = inputs.shape[0] // batch_size
        remaining_batch = inputs.shape[0] % batch_size
        outputs = np.zeros(shape=[inputs.shape[0]]+ list(self.module(inputs[:1]).shape[1:]))
        jitted_module = jax.jit(self.module.__call__)
        for batch_idx in range(num_batches):
            starting_idx = batch_idx * batch_size
            ending_idx = (batch_idx+1) * batch_size
            outputs[starting_idx:ending_idx] = jitted_module(inputs[starting_idx:ending_idx])
        if remaining_batch != 0:
            starting_idx = num_batches * batch_size
            ending_idx = inputs.shape[0]
            outputs[starting_idx:ending_idx] = self.module.__call__(inputs[starting_idx:ending_idx])
        return outputs
    
    def predict_loader(self, loader, max_batches=np.inf, verbose=1):
        from torch.utils.data import DataLoader
        if isinstance(loader, DataLoader):
            data = loader.dataset.__getitem__(0)
            try:
                x, y = data
            except:
                x = data
            x = np.array(x)[None]
            batch_size: int = loader.batch_size
            total_batch = min(len(loader), max_batches) * batch_size
        else:
            raise NotImplementedError()
        
        
        jitted_module = jax.jit(self.module.__call__)
        first_batch_output = jitted_module(x)
        list_output = isinstance(first_batch_output, list)
        if list_output:
            outputs_shape = [[total_batch] + list(out.shape[1:]) for out in first_batch_output]
            outputs = [np.zeros(shape) for shape in outputs_shape]
        else:
            outputs_shape = [total_batch] + list(first_batch_output.shape[1:])
            outputs = np.zeros(outputs_shape)
        
        starting_idx = 0
        enumerate_loader = enumerate(loader)
        if verbose >= 1:
            import tqdm
            enumerate_loader = tqdm.tqdm(enumerate_loader)
        for batch_idx, (inputs, _ )in enumerate_loader:
            inputs = np.array(inputs)
            if batch_idx >= max_batches:
                break
            batch_size = inputs.shape[0]
            ending_idx = starting_idx + batch_size
            if list_output:
                for i, output in enumerate(jitted_module(inputs)):
                    outputs[i][starting_idx:ending_idx] = output
            else:
                outputs[starting_idx:ending_idx] = jitted_module(inputs)
            starting_idx = ending_idx
        return outputs

    def evaluate_loader(self, loader, eval_fn, max_batches=np.inf, return_src=False, return_dict=False, verbose=1):
        from torch.utils.data import DataLoader
        if isinstance(loader, DataLoader):
            for data in loader:
                break
            try:
                x, _ = data
            except:
                x = data
            x = np.array(x)
            batch_size: int = loader.batch_size
            total_batch = min(len(loader), max_batches) * batch_size
        else:
            raise NotImplementedError()
        
        jitted_module = jax.jit(self.module.__call__)
        first_batch_output = jitted_module(x)
        list_output = isinstance(first_batch_output, list)
        
        if list_output:
            outputs_shape = [[total_batch] + list(out.shape[1:]) for out in first_batch_output]
            predictions = [np.zeros(shape) for shape in outputs_shape]
        else:
            outputs_shape = [total_batch] + list(first_batch_output.shape[1:])
            predictions = np.zeros(outputs_shape)
        labels = []
        
        enumerate_loader = enumerate(loader)
        if verbose >= 1:
            import tqdm
            enumerate_loader = tqdm.tqdm(enumerate_loader)
            
        starting_idx = 0
        for batch_idx, (inputs, targets) in enumerate_loader:
            inputs = np.array(inputs)
            ending_idx = starting_idx + inputs.shape[0]
            if batch_idx >= max_batches:
                break
            if list_output:
                for i, output in enumerate(jitted_module(inputs)):
                    predictions[i][starting_idx:ending_idx] = output
            else:
                predictions[starting_idx:ending_idx] = jitted_module(inputs)
            labels.append(targets)
            starting_idx = ending_idx

        evaluation_result = eval_fn(predictions, labels)
        if return_src:
            evaluation_result = (evaluation_result, x, labels, predictions)
        if return_dict:
            keys = ['scores'] 
            if return_src:
                keys = keys + ['x', 'targets', 'predictions']
            evaluation_result = {k: v for k, v in zip(keys, evaluation_result)}
        return evaluation_result

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
    
    def __init__(self, name=None):
        self.submodules = dict()
        self.taken_names_ = []
        self.params_ = VariablesDict()
        self.states_ = VariablesDict()
        self.trainable = True
        
        self.name = name if name is not None else UniqueID('Module')
        self.num_submodules = None
        self.num_params = None
        self.num_states = None
    
    def _post_init(self):
        for (attr_name, obj) in self.__dict__.items():
            if (hasattr(obj, 'is_collectable') and obj.is_collectable):
                self.submodules[attr_name] = obj
                obj.name = attr_name
                
        self.num_submodules = len(self.submodules)
    
    def __call__(self, inputs):
        if self.trainable:
            return self.call(inputs)
        return jax.lax.stop_gradient(self.call(jax.lax.stop_gradient(inputs)) )
    
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
            
    def param(self, name: str, shape=None, initializer=None, regularizer=None, value=None):
        """Manages the parameters of a model."""
        if shape is not None and (initializer is not None or value is not None):
            if name in self.taken_names_:
                raise ValueError(f"Name {name} is already taken.")
            
            self.taken_names_.append(name)
            self.params_[name] = Variable(shape, initializer, regularizer, value)
        elif name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")

        return self.params_.get_value(name)
    
    def state(self, name: str, shape=None, initializer=None, value=None):
        """Manages the states of a model."""
        if not (shape is None and initializer is None and value is None):
            if name in self.taken_names_:
                raise ValueError(f"Name {name} is already taken.")
            # initializer = init.NoInitializer() if initializer is None else initializer
            
            self.taken_names_.append(name)
            self.states_[name] = Variable(shape, initializer, None, value)
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
        super().__init__(name=UniqueID('Sequential'))
        if modules is None:
            modules = []
        else:
            # if not all(isinstance(module, Module) for module in modules):
            #     raise ValueError("All elements in 'modules' must be instances of 'Module'")
            modules = list(modules)
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
        self.submodules[module.name] = module
    
    def summary(self, input_shape):
        shape = input_shape
        self.summary_ = 'name shape num_parameters'
        self.summary_ = '{} {} {}\n'.format('input', shape, 0)
        for name in self.submodules:
            x = self.submodules[name](jnp.zeros(shape))
            shape = x.shape
            num_params = [name for name in self.params_.keys()]
            self.summary_ += '{} {} {}\n'.format(name, shape, num_params)
        return self.summary_
        

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
        super().__init__(name=UniqueID('Dense'))
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
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        kernel_reg,
        bias_reg,
        use_bias, 
        name,
        bias_shape,
        bias_init,
        dimension_numbers,
        dtype='float32'
    ):
        super().__init__(name=name)
        lhs_dilation = (1, 1) if lhs_dilation is None else lhs_dilation
        rhs_dilation = (1, 1) if rhs_dilation is None else rhs_dilation
        self.window_strides = window_strides if isinstance(window_strides, tuple) else (window_strides, window_strides) 
        self.lhs_dilation = lhs_dilation if isinstance(lhs_dilation, tuple) else (lhs_dilation, lhs_dilation)
        self.rhs_dilation = rhs_dilation if isinstance(rhs_dilation, tuple) else (rhs_dilation, rhs_dilation)
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.dimension_numbers = dimension_numbers
        if isinstance(padding, str):
            padding = padding.upper()
            if not padding in ['SAME', 'SAME_LOWER', 'VALID']:
                warnings.warn(f"Unsupported padding type: {padding}. Using 'SAME' as default.")
                padding = 'SAME'
            self.padding = padding
        else:
            self.padding = padding
            
        self.param('kernel', kernel_shape, kernel_init, kernel_reg)
        if self.use_bias:
            self.param('bias', bias_shape, bias_init, bias_reg)
    
    def call(self, x):
        x = lax.conv_general_dilated(x, self.param('kernel'), 
                                     window_strides = self.window_strides, 
                                     padding = self.padding,
                                     lhs_dilation = self.lhs_dilation, 
                                     rhs_dilation = self.rhs_dilation,
                                     dimension_numbers = self.dimension_numbers,
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
            kernel_init: init.Initializer=init.HeNormal(),
            bias_init=init.Zeros(),
            kernel_reg=None,
            bias_reg=None,
            dtype='float32'
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_height, kernel_width = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
        kernel_shape = (kernel_height, kernel_width, in_channels, out_channels)
        self.use_bias = use_bias
        bias_shape = (out_channels,) if self.use_bias else None
        
        super().__init__(kernel_shape, kernel_init,
                        feature_group_count=1,
                        window_strides=strides,
                        padding=padding,
                        lhs_dilation=None,
                        rhs_dilation=dilation,
                        kernel_reg=kernel_reg,
                        bias_reg=bias_reg,
                        use_bias=use_bias,
                        bias_shape=bias_shape,
                        bias_init=bias_init,
                        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                        name=UniqueID('Conv2d'),
                        dtype=dtype
                         )

class Conv2dTransposed(Module):
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
        super().__init__(name=UniqueID('Conv2dTransposed'))
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        kernel_shape = kernel_size + (self.out_channels, self.in_channels, )
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = padding.upper() if isinstance(padding, str) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dtype = dtype
        self.param('kernel', kernel_shape, kernel_init)
        self.param('bias', (out_channels, ), bias_init)
    

    def call(self, x):
        x = lax.conv_transpose(x, self.param('kernel'), 
                                strides = self.strides, 
                                padding = self.padding,
                                rhs_dilation = self.dilation,
                                transpose_kernel=True,
                                dimension_numbers = ('NHWC', 'HWIO', 'NHWC'))
        if self.use_bias:
            x = x + self.param('bias')
        return x

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
        name,
        eps=1e-6
    ):
        super().__init__(name=name)
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
        super().__init__(num_features, momentum, 0, running_mean_init, running_var_init, offset_init, scale_init, UniqueID('BatchNorm1d'), eps)
        
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
        super().__init__(num_features, momentum, (0, 1, 2), running_mean_init, running_var_init, offset_init, scale_init, UniqueID('BatchNorm2d'), eps)

class Dropout(Module):
    def __init__(self, drop_rate=0.5, seed=None):
        super().__init__()
        if seed is None:
            warnings.warn('Dropout seed is not provided. Proceed with default seed. Be careful with randomness.')
            seed = 42
        if not (0 <= drop_rate <= 1):
            warnings.warn("drop_rate should be in range [0, 1]. automatically rounded.")
            drop_rate = jnp.maximum(0.0, jnp.minimum(1, drop_rate))
        
        self.state('drop_rate', shape=(), initializer=init.ValueInitializer(drop_rate))
        self.state('seed', shape=(), initializer=init.ValueInitializer(seed))
    
    def call(self, inputs):
        if self.trainable:
            key, key_mask, key_seed = jax.random.split(jax.random.key(self.state('seed')), 3)
            
            keep_rate = 1 - self.state('drop_rate')            
            mask = jax.random.bernoulli(key_mask, keep_rate, inputs.shape)
            self.set_value('seed', jax.random.randint(key_seed, (), 0, MAX_UINT32 // 2) )
            return inputs * mask / (keep_rate+1e-8)
        return inputs

class DropBlock(Module):
    def __init__(self, block_size, drop_rate=0.05, seed=None):
        super().__init__()
        if seed is None:
            warnings.warn('DropBlock seed is not provided. Proceed with default seed. Be careful with randomness.')
            seed = 42
        if not (0 <= drop_rate <= 1):
            warnings.warn("drop_rate should be in range [0, 1]. automatically rounded.")
            drop_rate = jnp.maximum(0.0, jnp.minimum(1, drop_rate))
        
        self.state('drop_rate', shape=(), initializer=init.ValueInitializer(drop_rate))
        self.state('seed', shape=(), initializer=init.ValueInitializer(seed))
        self.block_size = (block_size, block_size) if isinstance(block_size, int) else block_size
        
    def call(self, inputs: jax.Array):
        if self.trainable:
            key, key_mask, key_seed = jax.random.split(jax.random.key(self.state('seed')), 3)
            
            keep_rate = 1 - self.state('drop_rate')  
            feat_size_square = inputs.shape[1] * inputs.shape[2]
            block_size_square = self.block_size[0] * self.block_size[1]
            
            gamma = (self.state('drop_rate')) / block_size_square * feat_size_square / (inputs.shape[1] - self.block_size[0] + 1) / (inputs.shape[2] - self.block_size[1] + 1)
                      
            mask = jax.random.bernoulli(key_mask, gamma, inputs.shape)
            mask = lax.reduce_window(mask, False, jnp.logical_or, (1,)+self.block_size+(1,), (1, 1, 1, 1), 'SAME')
            self.set_value('seed', jax.random.randint(key_seed, (), 0, MAX_UINT32 // 2) )
            return inputs * ~mask
        return inputs
    
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
        summed_feature = lax.reduce_window(inputs, 0.0, lax.add, self.kernel_size, self.strides, self.padding)
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