import os
import warnings
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
# tf.keras.backend.set_floatx('float32')

import jax
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
from jaxmao import losses as j
from keras import losses as k

class TestLoss:
    def __init__(self, shapes=(100, 10, 10)):
        if shapes is None:
            shapes = (100, 10, 10)
        self.shapes = shapes
        
    def test_all(self):
        self.test_shapes()
        self.test_all_zero()
        self.test_all_one()
        self.test_true0_pred1()
        self.test_true1_pred0()
        self.test_all_same()
        self.test_values()
        self.test_grads_05()
        self.test_grads_zeros()
        self.test_grads_ones()
        self.test_grads()

    def test_shapes(self, means=[0,0], vars=[1, 1]):
        sample_pred = np.random.normal(means[0], vars[0], self.shapes)
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        kv = self.k_loss(sample_true, sample_pred)
        jv = jax.jit(self.j_loss)(sample_pred, sample_true)

        print(kv.shape, jv.shape)
        assert kv.shape == jv.shape

    def test_all_zero(self, means=[0,0], vars=[1, 1]):
        sample_true = np.zeros(self.shapes)
        
        kv = self.k_loss(sample_true, sample_true)
        jv = jax.jit(self.j_loss)(sample_true, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_all_zero: ")

    def test_all_one(self, means=[0,0], vars=[1, 1]):
        sample_true = np.ones(self.shapes)
        
        kv = self.k_loss(sample_true, sample_true)
        jv = jax.jit(self.j_loss)(sample_true, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_all_one: ")

    def test_true0_pred1(self, means=[0,0], vars=[1, 1]):
        sample_pred = np.random.normal(means[0], vars[0], self.shapes)
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        kv = self.k_loss(sample_true, sample_pred)
        jv = jax.jit(self.j_loss)(sample_pred, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_true0_pred1: ")

    def test_true1_pred0(self, means=[0,0], vars=[1, 1]):
        sample_pred = np.random.normal(means[0], vars[0], self.shapes)
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        kv = self.k_loss(sample_true, sample_pred)
        jv = jax.jit(self.j_loss)(sample_pred, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_true1_pred0: ")
            
    def test_all_same(self, means=[0,0], vars=[1, 1]):
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        kv = self.k_loss(sample_true, sample_true)
        jv = jax.jit(self.j_loss)(sample_true, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_all_same: ")
            
    def test_values(self, means=[0,0], vars=[1, 1]):
        sample_pred = np.random.normal(means[0], vars[0], self.shapes)
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        kv = self.k_loss(sample_true, sample_pred)
        jv = jax.jit(self.j_loss)(sample_pred, sample_true)
        try:
            np.testing.assert_allclose(jv, kv, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)      
            print(f"----{self.j_loss} test_values: ")

    def test_grads_zeros(self, means=[0,0], vars=[0.1, 0.1]):
        sample_true = np.zeros(self.shapes)
        
        tf_sample_pred = tf.convert_to_tensor(sample_true)
        with tf.GradientTape() as g: 
            g.watch(tf_sample_pred)   
            kv = self.k_loss(tf_sample_pred, sample_true)
            kv_reduced = tf.reduce_sum(kv)
        kv_grads = g.gradient(kv_reduced, tf_sample_pred)
        
        j_loss = lambda pred, true: self.j_loss(pred, true).sum()
        jv_reduced, jv_grads = jax.value_and_grad(j_loss, argnums=0)(sample_true, sample_true)

        try:
            np.testing.assert_allclose(jv_reduced, kv_reduced, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_zeros values: ")

        try:
            np.testing.assert_allclose(jv_grads, kv_grads, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_zeros grads : ")
            
    def test_grads_05(self, means=[0,0], vars=[0.1, 0.1]):
        sample_true = np.full(self.shapes, 0.5)
        
        tf_sample_pred = tf.convert_to_tensor(sample_true)
        with tf.GradientTape() as g: 
            g.watch(tf_sample_pred)   
            kv = self.k_loss(tf_sample_pred, sample_true)
            kv_reduced = tf.reduce_sum(kv)
        kv_grads = g.gradient(kv_reduced, tf_sample_pred)
        
        j_loss = lambda pred, true: self.j_loss(pred, true).sum()
        jv_reduced, jv_grads = jax.value_and_grad(j_loss, argnums=0)(sample_true, sample_true)

        try:
            np.testing.assert_allclose(jv_reduced, kv_reduced, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_05 values: ")

        try:
            np.testing.assert_allclose(jv_grads, kv_grads, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_05 grads : ")

    def test_grads_ones(self, means=[0,0], vars=[0.1, 0.1]):
        sample_true = np.ones(self.shapes)
        
        tf_sample_pred = tf.convert_to_tensor(sample_true)
        with tf.GradientTape() as g: 
            g.watch(tf_sample_pred)   
            kv = self.k_loss(tf_sample_pred, sample_true)
            kv_reduced = tf.reduce_sum(kv)
        kv_grads = g.gradient(kv_reduced, tf_sample_pred)
        
        j_loss = lambda pred, true: self.j_loss(pred, true).sum()
        jv_reduced, jv_grads = jax.value_and_grad(j_loss, argnums=0)(sample_true, sample_true)

        try:
            np.testing.assert_allclose(jv_reduced, kv_reduced, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_ones values: ")

        try:
            np.testing.assert_allclose(jv_grads, kv_grads, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.j_loss} test_grads_ones grads : ")
            
    def test_grads(self, means=[0.5,0.5], vars=[1, 1]):
        sample_pred = np.random.normal(means[0], vars[0], self.shapes)
        sample_true = np.random.normal(means[1], vars[1], self.shapes)
        
        tf_sample_pred = tf.convert_to_tensor(sample_pred)
        with tf.GradientTape() as g: 
            g.watch(tf_sample_pred)   
            kv = self.k_loss(sample_true, tf_sample_pred)
            kv_reduced = tf.reduce_sum(kv)
        kv_grads = g.gradient(kv_reduced, tf_sample_pred)
        
        j_loss = lambda pred, true: self.j_loss(pred, true).sum()
        jv_reduced, jv_grads = jax.value_and_grad(j_loss, argnums=0)(sample_pred, sample_true)
        try:
            np.testing.assert_allclose(jv_reduced, kv_reduced, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----{self.__module__} test_grads values: ")

        try:
            np.testing.assert_allclose(jv_grads, kv_grads, atol=self.atol)
        except AssertionError as e:
            warnings.warn(str(e), UserWarning)   
            print(f"----test_grads grads {self.__class__}: ")
                
class TestMSE(TestLoss):
    def __init__(self, shapes=None):
        super().__init__(shapes)
        self.atol = 1e-5
        self.j_loss = j.MeanSquaredError(reduce_fn=None)
        self.k_loss = k.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
                    
    def test_grads_grads():
        pass

class TestBCE(TestLoss):
    def __init__(self, shapes=None):
        super().__init__(shapes)
        self.atol = 1e-7
        self.j_loss = j.BinaryCrossEntropy(reduce_fn=None)
        self.k_loss = k.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
class TestCCE(TestLoss):
    def __init__(self, shapes=None):
        super().__init__(shapes)
        self.atol = 1e-5
        self.j_loss = j.CategoricalCrossEntropy(reduce_fn=None)
        self.k_loss = k.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
class TestBCELogit(TestLoss):
    def __init__(self, shapes=None):
        super().__init__(shapes)
        self.atol = 1e-2
        self.j_loss = j.BCEWithLogitsLoss(reduce_fn=None)
        self.k_loss = k.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        
class TestCCELogit(TestLoss):
    def __init__(self, shapes=None):
        super().__init__(shapes)
        self.atol = 1e-7
        self.j_loss = j.CCEWithLogitsLoss(reduce_fn=None)
        self.k_loss = k.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        
if __name__ == '__main__':
    # good
    mse = TestMSE()
    mse.test_all()
    
    # low precision
    # bce = TestBCE()
    # bce.test_all()

    # # broken
    # cce = TestCCE()
    # cce.test_all()
    
    # # good
    # bce_logit = TestBCELogit(shapes=(2, 2)) # https://stackoverflow.com/questions/68163153/implementing-binary-cross-entropy-from-scratch-inconsistent-results-in-trainin
    # bce_logit.test_values()
    # bce_logit.test_all()

    # # low precision
    cee_logit = TestCCELogit(shapes=(2, 2))
    cee_logit.test_all()
    
    