from jax import jit
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

class optimizers:
    def __init__(self, params=None, lr=0.01):
        pass
            
    def __call__(self, params, gradients, states):
        return self.step(params, gradients, states)
        
class GradientDescent(optimizers):
    def __init__(self, params=None, lr=0.01):
        super().__init__()
        self.states = {'lr': lr}
        
    def step(self, params, gradients, states):
        return tree_map(lambda p, g : p - g*states['lr'], params, gradients), states
    
class Adam(optimizers):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.states = {
            'lr' : lr,
            'beta1' : beta1,
            'beta2' : beta2,
            't' : 0,
            'm' : tree_map(lambda x: jnp.zeros_like(x), params),
            'v' : tree_map(lambda x: jnp.zeros_like(x), params)
        }
        

    def step(self, params, gradients, states):
        lr    = states['lr']
        beta1 = states['beta1']
        beta2 = states['beta2']
        t_in     = states['t']
        m_in     = states['m']
        v_in     = states['v']
        
        t = t_in + 1
        m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, m_in, gradients)
        v = tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.power(g,2), v_in, gradients)
        
        m_corrected = tree_map(lambda m: m / (1 - jnp.power(beta1, t)), m)
        v_corrected = tree_map(lambda v: v / (1 - jnp.power(beta2, t)), v)
        updated_params =  tree_map(
                            lambda p, m_corr, v_corr : p - lr * m_corr / (jnp.sqrt(v_corr) + self.eps),
                            params, m_corrected, v_corrected
                            )
        updated_states = {'lr' : lr, 'beta1' : beta1, 'beta2' : beta2, 't' : t, 'm' : m, 'v' : v}
        return updated_params, updated_states
    
