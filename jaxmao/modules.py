from jaxmao.metrics import Metric
from jax import random, jit, value_and_grad
from sklearn.utils import shuffle
import numpy as np

class InitializeLayersModule(type):
    """
        Help Module class initialize layers without having to explicitly declared.
    """
    def __call__(cls, *args, **kwargs):
        instance = super(InitializeLayersModule, cls).__call__(*args, **kwargs)
        instance.post_initialization()
        return instance

class Module:
    def __init__(self):
        self.layers = {}
        self.state = {}
        self.params = {}
        self.loss = None
        self.optimizer = None
        self.metrics = None
        
    def post_initialization(self):
        self.init_layers()
        
    def init_layers(self):
        self.layers = dict()
        self.params = dict()
        for i, layer in enumerate(self.__dict__.values()):
            if isinstance(layer, self.classes_that_have_params):
                self.layers.append(layer)
        self.num_layers = len(self.layers)    
    
    def init_params(self, key):            
        for name in self.layers:
            key, subkey = random.split(key)
            self.layers[name].init_params(subkey)
            self.params[name] = self.layers[name].params
        self.num_layers = len(self.layers)
        return self.params
    
    def add(self, name, layer):
        self.layers[name] = layer
        if hasattr(layer, 'state'):
            self.state[name] = layer.state
    
    def forward(self, params, x, state):
        raise NotImplementedError("The forward method should be overridden by subclass")

    def pure_forward(self, params, x, state):
        """A pure function for the forward pass, to be used with JAX transformations."""
        out, new_state = self.forward(params, x, state)
        return out, new_state
    
    def __call__(self, params, x):
        out, new_state = self.forward(params, x, self.state)
        self.state.update(new_state)
        return out

    def forward_with_state(self, params, x, name, state):
        if name in self.layers:
            layer = self.layers[name]
            if isinstance(params, dict):
                if name in params:
                    x, layer_state = layer(params[name], x)
                    state[name] = layer_state
                else:
                    x, layer_state = layer(params, x)
                    state = layer_state
        return x, state
    
    def update_state(self, new_state):
        for name, layer in self.layers.items():
            if isinstance(layer.state, dict):
                layer.state.update(new_state[name])
                self.state[name] = layer.state

    def compile(
        self, loss_fn, optimizer, metrics=None
    ):
        if loss_fn:
            self.loss_fn = loss_fn
        if optimizer:
            self.optimizer = optimizer
        if metrics:
            if isinstance(metrics, dict):
                self.metrics = metrics
            elif isinstance(metrics, list):
                self.metrics = {metric.name if metric.name else f'metric_{n}': metric for n, metric in enumerate(self.metrics)}
            elif isinstance(metrics, Metric):
                self.metrics = {metrics.name: metrics}
        else:
            self.metrics = dict()
        
    def fit(
        self,
        X, y, lr=0.01, epochs=1, batch_size=32,
        X_val=None, y_val=None
            ):
        def aux_loss_fn(
            method, params, state,
            X, y, loss_fn, metrics
                ):
            # not pure because of model
            y_pred, new_state = jit(method)(params, X, state)
            
            # calculate metrics
            metric_values = dict()
            for metric_name, metric in metrics.items():
                metric_value = metric(y_pred, y)
                metric_values[metric_name] = metric_value
                
            # calculate loss
            loss = jit(loss_fn.calculate_loss)(y_pred, y)
            return loss, (new_state, metric_values)
        
        loss_and_grad = value_and_grad(aux_loss_fn, argnums=1, has_aux=True)
        
        optim_state = None
        num_batch = len(X) // batch_size 
        for epoch in range(epochs):
            losses = 0.0
            epoch_metrics = {name: 0.0 for name in self.metrics.keys()}  # Initialize metrics for the epoch

            X, y = shuffle(X, y)
            for n in range(num_batch):
                batch_x = X[n*batch_size:(n+1)*batch_size]
                batch_y = y[n*batch_size:(n+1)*batch_size]
                (loss, (new_state, batch_metrics)), gradients = loss_and_grad(
                                                            self.pure_forward, 
                                                            self.params, 
                                                            self.state, 
                                                            batch_x, 
                                                            batch_y, 
                                                            self.loss_fn,
                                                            self.metrics
                                                                )
                losses += loss
                for name, value in batch_metrics.items():
                    epoch_metrics[name] += value  # Aggregate metric values
                self.params, optim_state = self.optimizer.step(self.params, gradients, lr, optim_state)
                self.update_state(new_state)
            
            for name in epoch_metrics.keys():
                epoch_metrics[name] /= num_batch
            msg_metric = ' '.join([f"{name}: {value}" for name, value in epoch_metrics.items()])
            print(f'epoch {epoch}: Loss {losses/num_batch}; {msg_metric}')
