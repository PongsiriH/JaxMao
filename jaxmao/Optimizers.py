from jax import grad
from jax.tree_util import tree_map

class Optimizer:
    def __call__(self, params, gradients, lr=0.01):
        return self.step(params, gradients, lr=lr)
        
class GradientDescent(Optimizer):
    def step(self, params, gradients, lr=0.01):
        return tree_map(lambda p, g : p-lr*g, params, gradients)