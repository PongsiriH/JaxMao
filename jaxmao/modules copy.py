from jaxmao.layers import Layer
from jaxmao.metrics import Metric
from jax import random, jit, value_and_grad
from sklearn.utils import shuffle
import numpy as np

def compiled_loss_and_grad(method_fn, params, state, X, y, loss_fn, metrics):
    y_pred, new_state = method_fn(params, X, state)
    
    # calculate metrics
    metric_values = {name: metric(y_pred, y) for name, metric in metrics.items()}
    
    # calculate loss
    loss = loss_fn(y_pred, y)
    return loss, (new_state, metric_values)

loss_and_grad = value_and_grad(compiled_loss_and_grad, argnums=1, has_aux=True)


class InitializeLayersModule(type):
    """
        Help Module class initialize layers without having to explicitly declared.
    """
    def __call__(cls, *args, **kwargs):
        instance = super(InitializeLayersModule, cls).__call__(*args, **kwargs)
        instance.post_initialization()
        return instance

class Module(InitializeLayersModule):
    def __init__(self):
        self.layers = {}
        self.state = {}
        self.params = {}
        self.loss = None
        self.optimizer = None
        self.metrics = dict()
        self.num_params = 0
        self.training = False
        
        self.pure_forward = jit(self._pure_forward)
        
    def post_initialization(self):
        self.init_layers()
        
    def init_layers(self):
        self.layers = {}
        self.params = {}
        for i, (attr_name, layer) in enumerate(self.__dict__.items()):
            if isinstance(layer, Layer):
                self.layers[attr_name] = layer
                self.params[attr_name] = layer.params
        self.num_layers = len(self.layers)

    
    def init_params(self, key):            
        for name in self.layers:
            key, subkey = random.split(key)
            self.layers[name].init_params(subkey)
            self.params[name] = self.layers[name].params
        self.num_layers = len(self.layers)
        return self.params
    
    def forward(self, params, x, state):
        raise NotImplementedError("The forward method should be overridden by subclass")
    
    def __call__(self, x):
        out, new_state = self.forward(self.params, x, self.state)
        return out
    
    def update_state(self, new_state):
        self.state.update(new_state)
        for name, layer in self.layers.items():
            layer.state.update(new_state[name])

    def set_training_mode(self):
        self.training = True
        self.pure_forward = jit(self._pure_forward)
        for layer in self.layers.values():
            layer.set_training_mode()

    def set_inference_mode(self):
        self.training = False
        self.pure_forward = jit(self._pure_forward)
        for layer in self.layers.values():
            layer.set_inference_mode()