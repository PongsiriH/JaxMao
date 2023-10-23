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

class Module:
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

    def _pure_forward(self, params, x, state):
        """A pure function for the forward pass, to be used with JAX transformations."""
        out, new_state = self.forward(params, x, state)
        return out, new_state
    
    def __call__(self, params, x):
        out, new_state = self.forward(params, x, self.state)
        self.state.update(new_state)
        return out

    def apply(self, params, x, name, state):
        return self.forward_with_state(params, x, name, state)
        # z, new_state = self.layers[name].pure_forward(params, x, state)
        # self.update_state(new_state)
        # return z, new_state

    def forward_with_state(self, params, x, name, state):
        if name in self.layers:
            layer = self.layers[name]
            if isinstance(params, dict):
                if name in params:
                    x, layer_state = layer(params[name], x)
                    state[name].update(layer_state)
                else:
                    x, layer_state = layer(params, x)
                    state.update(layer_state)
        return x, state

    def forward_build(self, params, x, name, state):
        if name in self.layers:
            layer = self.layers[name]
            if isinstance(params, dict):
                if name in params:
                    x, layer_state = layer(params[name], x)
                    state[name].update(layer_state)
                else:
                    x, layer_state = layer(params, x)
                    state.update(layer_state)
            self.summary += '{:<20} {:<20} {:<20} {:<20}\n'.format(name, str(x.shape), layer.count_params(), layer.num_states)
            self.num_params += layer.num_params
        return x, state
    
    def update_state(self, new_state):
        self.state.update(new_state)
        for name, layer in self.layers.items():
            layer.state.update(new_state[name])
            # self.state[name] = layer.state

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

    def compile(
        self, loss_fn, optimizer, metrics=None
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if metrics is not None:
            if isinstance(metrics, dict):
                self.metrics = metrics
            elif isinstance(metrics, list):
                self.metrics = {metric.name if hasattr(metric, 'name') else f'metric_{n}': 
                                metric for n, metric in enumerate(metrics)}
            elif isinstance(metrics, Metric):
                self.metrics = {metrics.name: metrics}
        else:
            self.metrics = dict()

    def fit(
        self,
        X, y, lr=0.01, epochs=1, batch_size=32,
        X_val=None, y_val=None
    ):
        num_batch = len(X) // batch_size
        
        header_metrics = ' '.join([f"{name: <15}" for name in self.metrics.keys()])
        print(f"{'epoch': <10}{'avg_loss': <15}{header_metrics}")
        for epoch in range(epochs):
            losses = 0.0
            epoch_metrics = np.zeros(len(self.metrics))

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
                epoch_metrics += np.array(list(batch_metrics.values()))
                
                self.params, self.optimizer.state = self.optimizer.step(self.params, gradients, self.optimizer.state)
                self.update_state(new_state)

            epoch_metrics /= num_batch
            losses_avg = losses/num_batch

            msg_metric = ' '.join([f"{value:<15.4}" for value in epoch_metrics])
            msg_val = ' '
            if X_val is not None and y_val is not None:
                val_metric = self.eval(X_val, y_val)
                msg_val = msg_val.join([f'{value:<15.4}' for value in val_metric.values()])
                msg_val = '\n{:<10} '.format(' ') + msg_val
            print(f'{epoch:<10} {losses_avg: <15.4f}{msg_metric} {msg_val}\n')


    def eval(self, X, y):
        y_pred, state = self.pure_forward(self.params, X, self.state)
        metric_values = {'loss': self.loss_fn(y_pred, y)}
        metric_values.update({name: metric(y_pred, y) for name, metric in self.metrics.items()})
        return metric_values
    
    def summarize(self, input_shape):
        training = self.training
        self.set_inference_mode()
        self.apply = self.forward_build
        input_shape = np.array(input_shape)
        input_shape[0] = 4

        self.num_params = 0
        self.summary = '{:<20} {:<20} {:<20} {:<20}\n'.format('layer', 'output shape', '#\'s params', '#\'s states')
        out, state = self.forward(self.params, np.random.normal(0, 1, input_shape), self.state)
        
        msg = self.summary + f'\ntotal parameters: {self.num_params}'
        print(msg)
        
        if training:
            self.set_training_mode()
        self.apply = self.forward_with_state
        return msg