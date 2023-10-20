from jax import jit
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

class optimizers:
    def __init__(self):
        self.step = jit(self.step)
    def __call__(self, params, gradients, lr=0.01):
        return self.step(params, gradients, lr=lr)
        
class GradientDescent:
    def step(self, params, gradients, lr=0.01, state=None):
        return tree_map(lambda p, g : p - lr * g, params, gradients), state
    
class Adam(optimizers):
    def __init__(self, params=None, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.t = 0
        
        self.is_mt_Initialized = False
        if params:
            self.m = tree_map(lambda x: jnp.zeros_like(x), params)
            self.v = tree_map(lambda x: jnp.zeros_like(x), params)
            self.is_mt_Initialized = True
    @jit
    def step(self, params, gradients, lr=None, beta1=None, beta2=None, epsilon=1e-8):
        if not self.is_mt_Initialized:
            self.m = tree_map(lambda x: jnp.zeros_like(x), params)
            self.v = tree_map(lambda x: jnp.zeros_like(x), params)
            self.is_mt_Initialized = True
        if not lr:
            lr = self.lr
        if not beta1:
            beta1 = self.beta1
        if not beta2:
            beta2 = self.beta2
        
        self.t += 1
        self.m = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, self.m, gradients)
        self.v = tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.power(g,2), self.v, gradients)
        
        m_corrected = tree_map(lambda m: m / (1 - jnp.power(beta1, self.t)), self.m)
        v_corrected = tree_map(lambda v: v / (1 - jnp.power(beta2, self.t)), self.v)
        return tree_map(lambda p, m_corr, v_corr: p - lr * m_corr / (jnp.sqrt(v_corr) + epsilon),
                        params, m_corrected, v_corrected)
        

class Adam(optimizers):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = tree_map(lambda x: jnp.zeros_like(x), params)
        self.v = tree_map(lambda x: jnp.zeros_like(x), params)

    def step(self, params, gradients):
        self.t += 1
        self.m = tree_map(lambda m, g: self.beta1 * m + (1 - self.beta1) * g, self.m, gradients)
        self.v = tree_map(lambda v, g: self.beta2 * v + (1 - self.beta2) * jnp.power(g,2), self.v, gradients)
        
        m_corrected = tree_map(lambda m: m / (1 - jnp.power(self.beta1, self.t)), self.m)
        v_corrected = tree_map(lambda v: v / (1 - jnp.power(self.beta2, self.t)), self.v)
        return tree_map(lambda p, m_corr, v_corr: p - self.lr * m_corr / (jnp.sqrt(v_corr) + self.epsilon),
                        params, m_corrected, v_corrected)