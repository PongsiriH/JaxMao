from jax import jit
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

class optimizers:
    def __init__(self):
        self.state = dict({'lr': 0.01})
        self.step = jit(self.step)
        
    def __call__(self, params, gradients, state):
        return self.step(params, gradients, state)
        
class GradientDescent(optimizers):
    def __init__(self, lr=0.01, params=None):
        super().__init__()
        self.state['lr'] = lr
        
    def step(self, params, gradients, state):
        return tree_map(lambda p, g : p - g*state['lr'], params, gradients), state
    
class Adam(optimizers):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-7):
        super().__init__()
        self.state = {
            'lr' : lr,
            'beta1' : beta1,
            'beta2' : beta2,
            't' : 0,
            'm' : tree_map(lambda x: jnp.zeros_like(x), params),
            'v' : tree_map(lambda x: jnp.zeros_like(x), params)
        }
        self.eps = eps

    def step(self, params, gradients, state):
        lr    = state['lr']
        beta1 = state['beta1']
        beta2 = state['beta2']
        t_in     = state['t']
        m_in     = state['m']
        v_in     = state['v']
        
        t = t_in + 1
        m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, m_in, gradients)
        v = tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.power(g,2), v_in, gradients)
        
        m_corrected = tree_map(lambda m: m / (1 - jnp.power(beta1, t)), m)
        v_corrected = tree_map(lambda v: v / (1 - jnp.power(beta2, t)), v)
        updated_params =  tree_map(
                            lambda p, m_corr, v_corr : p - lr * m_corr / (jnp.sqrt(v_corr) + self.eps),
                            params, m_corrected, v_corrected
                            )
        updated_state = {'lr' : lr, 'beta1' : beta1, 'beta2' : beta2, 't' : t, 'm' : m, 'v' : v}
        return updated_params, updated_state
    
