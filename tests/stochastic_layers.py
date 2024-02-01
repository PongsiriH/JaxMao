"""TODO

"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import random, gc, time, copy

from jaxmao import Bind, Dropout, DropBlock
import jax
import jax.numpy as jnp

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')
SEED = 42

class TestDropout:
    def __init__(self):
        self.atol = 1e-12
        self.key = jax.random.key(SEED)
        
    def setup(self, drop_rate=0.5, seed: int=SEED):
        key = jax.random.key(seed)
        dropout = Dropout(drop_rate, seed)
        params, states = dropout.init(key)
        return dropout, params, states
    
    def test_all(self):
        self.test_inference()
        self.test_drop_all()
        self.test_drop_none()
        self.test_dropout_stochasticity((10,), 0.5)
        self.test_dropout_stochasticity((10, 20, 10, 20), 0.9)
        self.test_dropout_stochasticity((50, 20), 0.2)
        self.test_dropout_scaling((10, 100, 10), 0.5)
        self.test_dropout_scaling((10,), 0.5)
        self.test_dropout_scaling((10, 20, 10, 20), 0.9)
        self.test_dropout_scaling((50, 20), 0.2)
    
    def test_inference(self):
        dropout, params, states = self.setup()
        with Bind(dropout, params, states, trainable=False) as ctx:
            inputs = jax.random.normal(self.key, (10, 100, 10))        
            outputs = ctx.module(inputs)
        assert inputs.shape == outputs.shape
        assert np.allclose(inputs, outputs, atol=self.atol)

    def test_drop_all(self):
        dropout, params, states = self.setup(1.0)
        with Bind(dropout, params, states, trainable=True) as ctx:
            inputs = jax.random.normal(self.key, (10, 100, 10))  
            outputs = ctx.module(inputs)
        assert inputs.shape == outputs.shape
        assert np.allclose(jnp.zeros_like(inputs), outputs, atol=self.atol)

    def test_drop_none(self):
        dropout, params, states = self.setup(0.0)
        with Bind(dropout, params, states, trainable=True) as ctx:
            inputs = jax.random.normal(self.key, (10, 100, 10))  
            outputs = ctx.module(inputs)

        assert inputs.shape == outputs.shape
        assert np.allclose(inputs, outputs, atol=self.atol)
    
    def test_dropout_stochasticity(self, input_shape, drop_rate, num_runs=100):
        active_counts = []
        
        for k in range(num_runs):
            dropout, params, states = self.setup(drop_rate, k)
            with Bind(dropout, params, states, trainable=True) as ctx:
                inputs = jax.random.normal(jax.random.PRNGKey(k), input_shape)
                outputs = ctx.module(inputs)
                active_count = np.sum(outputs != 0)
                active_counts.append(active_count)

        average_active_rate = np.mean(active_counts) / np.prod(input_shape)
        expected_active_rate = 1 - drop_rate
        assert np.isclose(average_active_rate, expected_active_rate, atol=0.05)    

    def test_dropout_scaling(self, input_shape, drop_rate):
        dropout, params, states = self.setup(drop_rate)
        with Bind(dropout, params, states, trainable=True) as ctx:
            inputs = jnp.ones(input_shape)
            outputs = ctx.module(inputs)
        scale_factor = 1.0 / (1 - drop_rate)
        
        active_mask = outputs != 0

        scaled_values_correct = np.allclose(outputs[active_mask], scale_factor, atol=0.01)
        assert scaled_values_correct

class TestDropBlock:
    def __init__(self):
        self.atol = 1e-12
        
    def test_inference(self):
        self.dropblock.set_trainable(False)
        inputs = jax.random.normal(self.key, (10, 100, 10))        
        outputs = self.dropout(inputs)

        assert inputs.shape == outputs.shape
        assert np.allclose(inputs, outputs, atol=self.atol)
        
if __name__ == '__main__':
    dropout = TestDropout()
    dropout.test_all()

    # dropblock = TestDropBlock()
    # dropblock.test_inference()