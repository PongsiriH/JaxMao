"""TODO

"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import random, gc, time

from jaxmao import Bind, Dropout
import jax
import jax.numpy as jnp

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')
SEED = 42

class TestDropout:
    def __init__(self):
        self.atol = 1e-12
        self.dropout = Dropout(0.5)
        self.key = jax.random.key(SEED)

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
        self.dropout.set_trainable(False)
        inputs = jax.random.normal(self.key, (10, 100, 10))        
        outputs = self.dropout(inputs)

        assert inputs.shape == outputs.shape
        assert np.allclose(inputs, outputs, atol=self.atol)

    def test_drop_all(self):
        self.dropout.set_trainable(True)
        inputs = jax.random.normal(self.key, (10, 100, 10))  
        dropout_1 = Dropout(1.0)      
        outputs = dropout_1(inputs)

        assert inputs.shape == outputs.shape
        assert np.allclose(jnp.zeros_like(inputs), outputs, atol=self.atol)

    def test_drop_none(self):
        self.dropout.set_trainable(True)
        inputs = jax.random.normal(self.key, (10, 100, 10))  
        dropout_1 = Dropout(0.0)      
        outputs = dropout_1(inputs)

        assert inputs.shape == outputs.shape
        assert np.allclose(inputs, outputs, atol=self.atol)
    
    def test_dropout_stochasticity(self, input_shape, drop_rate, num_runs=100):
        active_counts = []
        dropout_layer = Dropout(drop_rate)
        
        for _ in range(num_runs):
            inputs = jax.random.normal(jax.random.PRNGKey(_), input_shape)
            outputs = dropout_layer(inputs)
            active_count = np.sum(outputs != 0)
            active_counts.append(active_count)

        average_active_rate = np.mean(active_counts) / np.prod(input_shape)
        expected_active_rate = 1 - drop_rate
        assert np.isclose(average_active_rate, expected_active_rate, atol=0.05)    

    def test_dropout_scaling(self, input_shape, drop_rate):
        dropout_layer = Dropout(drop_rate)
        inputs = jnp.ones(input_shape)
        outputs = dropout_layer(inputs)
        scale_factor = 1.0 / (1 - drop_rate)
        
        active_mask = outputs != 0

        scaled_values_correct = np.allclose(outputs[active_mask], scale_factor, atol=0.01)
        assert scaled_values_correct

if __name__ == '__main__':
    dropout = TestDropout()
    dropout.test_all()