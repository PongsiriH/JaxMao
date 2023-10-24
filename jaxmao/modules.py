from typing import Tuple

from utils_struct import (
                    RecursiveDict,
                    PostInitialization
                          )

from metrics import Metric
from jax import random, jit, value_and_grad
from sklearn.utils import shuffle
import numpy as np


class Module(metaclass=PostInitialization):
    is_collectable = True
    
    def __init__(self):
        self.layers = {}
        self.state = RecursiveDict()
        self.params = RecursiveDict()
        
        self.num_params = 0
        self.num_states = 0

        self.training = False
        
        self.loss = None
        self.optimizer = None
        self.metrics = dict()
        
    def post_initialization(self):
        for (attr_name, obj) in self.__dict__.items():  # collect layers and params
            if hasattr(obj, 'is_collectable') and obj.is_collectable:
                self.layers[attr_name] = obj
                self.params[attr_name] = obj.params
                self.state[attr_name] = obj.state
                obj.name = attr_name
        self.num_layers = len(self.layers)
        self.pure_forward = jit(self.pure_forward)

    def init_params(self, key):            
        for name in self.layers:
            key, subkey = random.split(key)
            self.layers[name].init_params(subkey)
            self.params[name] = self.layers[name].params 

    def update_params(self, new_params):
        for name in self.layers:
            self.params[name] = new_params[name]

    def update_state(self, new_state):
        for name in self.layers:
            self.state[name] = new_state[name]
            
    def pure_forward(
                self, 
                params: RecursiveDict, 
                x, 
                state: RecursiveDict
                ) -> Tuple[RecursiveDict, RecursiveDict]:
        """
            Returns:
            - Tuple[RecursiveDict, RecursiveDict]: A tuple containing the output and the new state.
        """
        raise NotImplementedError("The forward method should be overridden by subclass. Keep in mind that forward must return tuple(f(x), new_state) be pure (JAX-wise).")

    def forward(self, layer, params, x, state):
        raise NotImplementedError("Please define pure_forward and switch_mode.")

    def _lazy_train_forward(self, layer, params, x, state):
        """forward particular layer."""
        x, new_state = layer.pure_forward(params[layer.name], x, state[layer.name])
        state[layer.name] = new_state
        return x, state

    def _lazy_inference_forward(self, layer, params, x, state):
        """forward particular layer but don't use params and do not update state."""
        x, new_state = layer.pure_forward(layer.params, x, state[layer.name])
        return x, state

    def __call__(self, x):
        out, new_state = self.pure_forward(self.params, x, self.state)
        return out
    
    def switch_mode(self, mode : str):
        mode = mode.lower()
        if mode == 'train':
            self.forward = self._lazy_train_forward
            self.training = True
            for layer in self.layers.values():
                layer.switch_mode('train')
        elif mode == 'inference':
            self.forward = self._lazy_inference_forward
            self.training = False
            for layer in self.layers.values():
                layer.switch_mode('inference')
        else:
            ValueError('Only accept values [\'train\', \'inference\'].')