from jax import grad
from jax.tree_util import tree_map

class Optimizers:
    pass
    pass
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.grad_loss = grad(loss_fn)
        
class GradientDescent(Optimizers):
    def step(self, params, gradients, lr=0.01):
        return tree_map(lambda p, g : p-lr*g, params, gradients)