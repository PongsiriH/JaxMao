import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import random, gc, time

from jaxmao.modules import Dense, Module, BatchNorm1d
import jax

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')

eps = 1e-6
from jaxmao import optimizers


# JaxMao model
class DenseBlock(Module):
    def __init__(self, in_channels, excited_size):
        super().__init__()
        self.squeeze = Dense(in_channels, excited_size)
        self.bn1 = BatchNorm1d(excited_size, eps=eps, momentum=0.5)
        self.excite = Dense(excited_size, in_channels)
        self.bn2 = BatchNorm1d(in_channels, eps=eps, momentum=0.5)
    def call(self, x):
        x = jax.nn.sigmoid(self.bn1(self.squeeze(x)))
        return self.bn2(self.excite(x))

class NestedDense(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense1 = DenseBlock(in_channels, 32)
        self.dense2 = DenseBlock(in_channels, out_channels)
        self.dense3 = Dense(in_channels, out_channels)
        
    def call(self, x):
        return self.dense3(self.dense2(self.dense1(x)))

# Keras model
class KerasDenseBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, excited_size):
        super(KerasDenseBlock, self).__init__()
        self.squeeze = tf.keras.layers.Dense(excited_size)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=eps)
        self.excite = tf.keras.layers.Dense(in_channels)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.5, epsilon=eps)

    def call(self, inputs):
        x = self.bn1(self.squeeze(inputs))
        x = tf.nn.sigmoid(x)
        x = self.bn2(self.excite(x))
        return x

class KerasNestedDense(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super(KerasNestedDense, self).__init__()
        self.dense1 = KerasDenseBlock(in_channels, 32)
        self.dense2 = KerasDenseBlock(in_channels, out_channels)
        self.dense3 = keras.layers.Dense(out_channels)
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense3(self.dense2(x))
    
class TestModule:
    def test_single_dense(self):
        in_channels = 22
        out_channels = 44
        atol = 1e-6
        
        class SingleDense(Module):
            def __init__(self):
                super().__init__()
                self.dense = Dense(in_channels, out_channels)

            def call(self, x):
                return jax.nn.softmax(self.dense(x))

        sample_data = np.random.normal(0, 1, (8, in_channels))
        
        jaxmao_model = SingleDense()
        params, states = jaxmao_model.init(jax.random.key(0))
        keras_model = keras.layers.Dense(out_channels, 'softmax')
        keras_model.build(input_shape=(None, in_channels))
        keras_model.set_weights([params['dense']['kernel'], params['dense']['bias']])
        
        jaxmao_prediction, states, _ = jaxmao_model.apply(sample_data, params, states)
        keras_prediction = keras_model(sample_data)

        assert jaxmao_prediction.shape == keras_prediction.shape
        # print(np.abs(jaxmao_prediction - keras_prediction.numpy()).max())
        assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=atol)
        
    def test_many_dense(self):
        in_channels = 22
        out_channels = 44
        atol = 1e-6
        
        class ManyDense(Module):
            def __init__(self):
                super().__init__()
                self.dense1 = Dense(in_channels, 24)
                self.dense2 = Dense(24, out_channels)

            def call(self, x):
                x = jax.nn.relu(self.dense1(x))
                return jax.nn.softmax(self.dense2(x))

        sample_data = np.random.normal(0, 1, (8, in_channels))
        
        jaxmao_model = ManyDense()
        params, states = jaxmao_model.init(jax.random.key(0))
        keras_model = keras.Sequential([keras.layers.Dense(24, 'relu'), keras.layers.Dense(out_channels, 'softmax')])
        keras_model.build(input_shape=(None, in_channels))
        keras_model.layers[0].set_weights([params['dense1']['kernel'], params['dense1']['bias']])
        keras_model.layers[1].set_weights([params['dense2']['kernel'], params['dense2']['bias']])
        
        jaxmao_prediction, states, _ = jaxmao_model.apply(sample_data, params, states)
        keras_prediction = keras_model(sample_data)

        assert jaxmao_prediction.shape == keras_prediction.shape
        assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=atol)
            
    def test_nested_dense(self):
        in_channels = 22
        out_channels = 44
        atol = 1e-6
        
        # JaxMao model
        class DenseBlock(Module):
            def __init__(self, in_channels, excited_size):
                super().__init__()
                self.squeeze = Dense(in_channels, excited_size)
                self.excite = Dense(excited_size, in_channels)
            def call(self, x):
                return self.excite(jax.nn.relu(self.squeeze(x)))

        class NestedDense(Module):
            def __init__(self):
                super().__init__()
                self.dense1 = DenseBlock(in_channels, 32)
                self.dense2 = DenseBlock(in_channels, out_channels)
            def call(self, x):
                return self.dense2(self.dense1(x))
        
        # Keras model
        class KerasDenseBlock(tf.keras.layers.Layer):
            def __init__(self, in_channels, excited_size):
                super(KerasDenseBlock, self).__init__()
                self.squeeze = tf.keras.layers.Dense(excited_size, activation='relu')
                self.excite = tf.keras.layers.Dense(in_channels)

            def call(self, inputs):
                x = self.squeeze(inputs)
                return self.excite(x)

        class KerasNestedDense(tf.keras.Model):
            def __init__(self, in_channels, out_channels):
                super(KerasNestedDense, self).__init__()
                self.dense1 = KerasDenseBlock(in_channels, 32)
                self.dense2 = KerasDenseBlock(in_channels, out_channels)

            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)
    
        sample_data = np.random.normal(0, 1, (8, in_channels))
        
        jaxmao_model = NestedDense()
        params, states = jaxmao_model.init(jax.random.key(0))
        keras_model = KerasNestedDense(in_channels, out_channels)
        keras_model.build(input_shape=(None, in_channels))

        keras_model.layers[0].squeeze.set_weights([params['dense1']['squeeze']['kernel'], params['dense1']['squeeze']['bias']])
        keras_model.layers[0].excite.set_weights([params['dense1']['excite']['kernel'], params['dense1']['excite']['bias']])

        keras_model.layers[1].squeeze.set_weights([params['dense2']['squeeze']['kernel'], params['dense2']['squeeze']['bias']])
        keras_model.layers[1].excite.set_weights([params['dense2']['excite']['kernel'], params['dense2']['excite']['bias']])

        
        jaxmao_prediction, states, _ = jaxmao_model.apply(sample_data, params, states)
        keras_prediction = keras_model(sample_data)

        assert jaxmao_prediction.shape == keras_prediction.shape
        assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=atol)    
         
class TestNestedDenseModules:
    def __init__(self):
        self.in_channels = 22
        self.out_channels = 44
        self.rtol = 1e-6
        self.atol = 1e-5
        self.eps = 1e-6

        self.jaxmao_gd = optimizers.GradientDescent()
        
        self.jaxmao_model = NestedDense(self.in_channels, self.out_channels)
        self.params, self.states = self.jaxmao_model.init(jax.random.key(42))
        
        self.keras_model = KerasNestedDense(self.in_channels, self.out_channels)
        self.keras_model.build(input_shape=(None, self.in_channels))

        self.keras_model.layers[0].squeeze.set_weights([self.params['dense1']['squeeze']['kernel'], self.params['dense1']['squeeze']['bias']])
        self.keras_model.layers[0].excite.set_weights([self.params['dense1']['excite']['kernel'], self.params['dense1']['excite']['bias']])

        self.keras_model.layers[1].squeeze.set_weights([self.params['dense2']['squeeze']['kernel'], self.params['dense2']['squeeze']['bias']])
        self.keras_model.layers[1].excite.set_weights([self.params['dense2']['excite']['kernel'], self.params['dense2']['excite']['bias']])

        self.keras_model.layers[2].set_weights([self.params['dense3']['kernel'], self.params['dense3']['bias']])
        
    def test_nested_dense_with_batch_normalization(self):
        sample_data = np.random.uniform(2, 4, (128, self.in_channels))

        self.jaxmao_model.set_trainable(True)
        for num_loop in range(100):
            jaxmao_prediction, self.states, _ = self.jaxmao_model.apply(sample_data, self.params, self.states)
            keras_prediction = self.keras_model(sample_data, training=True)

            assert jaxmao_prediction.shape == keras_prediction.shape
            assert self.states['dense1']['bn1']['running_mean'].shape == self.keras_model.layers[0].bn1.moving_mean.numpy().shape 
            assert self.states['dense1']['bn1']['running_var'].shape == self.keras_model.layers[0].bn1.moving_variance.numpy().shape 
            assert self.states['dense1']['bn2']['running_mean'].shape == self.keras_model.layers[0].bn2.moving_mean.numpy().shape 
            assert self.states['dense1']['bn2']['running_var'].shape == self.keras_model.layers[0].bn2.moving_variance.numpy().shape 
            assert self.states['dense2']['bn1']['running_mean'].shape == self.keras_model.layers[1].bn1.moving_mean.numpy().shape 
            assert self.states['dense2']['bn1']['running_var'].shape == self.keras_model.layers[1].bn1.moving_variance.numpy().shape 
            assert self.states['dense2']['bn2']['running_mean'].shape == self.keras_model.layers[1].bn2.moving_mean.numpy().shape 
            assert self.states['dense2']['bn2']['running_var'].shape == self.keras_model.layers[1].bn2.moving_variance.numpy().shape 

            assert np.allclose(self.states['dense1']['bn1']['running_mean'], self.keras_model.layers[0].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn1']['running_var'], self.keras_model.layers[0].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_mean'], self.keras_model.layers[0].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_var'], self.keras_model.layers[0].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_mean'], self.keras_model.layers[1].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_var'], self.keras_model.layers[1].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_mean'], self.keras_model.layers[1].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_var'], self.keras_model.layers[1].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=self.atol)    
    
    def test_randomly_switch_trainable_on_nested_dense(self):
        sample_data = np.random.uniform(2, 4, (128, self.in_channels))

        for num_loop in range(100):
            trainable_status = np.random.choice([True, False])
            self.jaxmao_model.set_trainable(trainable_status)
            jaxmao_prediction, self.states, _ = self.jaxmao_model.apply(sample_data, self.params, self.states)
            keras_prediction = self.keras_model(sample_data, training=trainable_status)

            assert jaxmao_prediction.shape == keras_prediction.shape
            assert self.states['dense1']['bn1']['running_mean'].shape == self.keras_model.layers[0].bn1.moving_mean.numpy().shape 
            assert self.states['dense1']['bn1']['running_var'].shape == self.keras_model.layers[0].bn1.moving_variance.numpy().shape 
            assert self.states['dense1']['bn2']['running_mean'].shape == self.keras_model.layers[0].bn2.moving_mean.numpy().shape 
            assert self.states['dense1']['bn2']['running_var'].shape == self.keras_model.layers[0].bn2.moving_variance.numpy().shape 
            assert self.states['dense2']['bn1']['running_mean'].shape == self.keras_model.layers[1].bn1.moving_mean.numpy().shape 
            assert self.states['dense2']['bn1']['running_var'].shape == self.keras_model.layers[1].bn1.moving_variance.numpy().shape 
            assert self.states['dense2']['bn2']['running_mean'].shape == self.keras_model.layers[1].bn2.moving_mean.numpy().shape 
            assert self.states['dense2']['bn2']['running_var'].shape == self.keras_model.layers[1].bn2.moving_variance.numpy().shape 

            assert np.allclose(self.states['dense1']['bn1']['running_mean'], self.keras_model.layers[0].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn1']['running_var'], self.keras_model.layers[0].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_mean'], self.keras_model.layers[0].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_var'], self.keras_model.layers[0].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_mean'], self.keras_model.layers[1].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_var'], self.keras_model.layers[1].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_mean'], self.keras_model.layers[1].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_var'], self.keras_model.layers[1].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=self.atol)     
    
    def test_randomly_switch_trainable_on_each_layer_of_nested_dense(self):
        pass
    
    def test_gradients(self):
        optimizer = self.jaxmao_gd
        self.jaxmao_optimizer_state = self.jaxmao_gd.states
        
        def keras_gradient_descent_step(model, input_data, target_data, learning_rate=0.01):
            with tf.GradientTape() as tape:
                predictions = model(input_data, training=True)
                loss = tf.reduce_mean(tf.square(predictions - target_data))

            gradients = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, gradients):
                var.assign_sub(learning_rate * grad)

        def jaxmao_gradient_descet_step(input_data, target_data, params, states, optimizer_state):
            def loss_fn(input_data, target_data, params, states):
                predictions, states, _= self.jaxmao_model.apply(input_data, params, states)
                return jax.numpy.mean(jax.numpy.square(predictions - target_data)), states
            (loss, states), gradients = jax.value_and_grad(loss_fn, argnums=2, has_aux=True)(input_data, target_data, params, states)
            params, optimizer_state = optimizer(params, gradients, optimizer_state)
            return loss, gradients, params, states, optimizer_state
        
        sample_data = np.random.uniform(2, 4, (128, self.in_channels))
        sample_labels = np.random.normal(5, 2, (128, self.out_channels))
            
        for num_loop in range(100):
            self.jaxmao_model.set_trainable(True)
            # Perform gradient descent step on JAX model
            loss_jax, gradients_jax, self.params, self.states, self.jaxmao_optimizer_state = jaxmao_gradient_descet_step(
                    sample_data, sample_labels,
                    params=self.params, states=self.states, optimizer_state=self.jaxmao_optimizer_state
                    )

            # Perform gradient descent step on Keras model
            keras_model = keras_gradient_descent_step(self.keras_model, sample_data, sample_labels, learning_rate=0.01)

            self.jaxmao_model.set_trainable(False)
            jaxmao_prediction, self.states, _= self.jaxmao_model.apply(sample_data, self.params, self.states)
            keras_prediction = self.keras_model(sample_data, training=False)
            self.jaxmao_model.set_trainable(True)
            
            assert jaxmao_prediction.shape == keras_prediction.shape
            
            assert self.params['dense1']['squeeze']['kernel'].shape == self.keras_model.layers[0].squeeze.kernel.numpy().shape 
            assert self.params['dense1']['squeeze']['bias'].shape == self.keras_model.layers[0].squeeze.bias.numpy().shape 
            assert self.params['dense1']['excite']['kernel'].shape == self.keras_model.layers[0].excite.kernel.numpy().shape 
            assert self.params['dense1']['excite']['bias'].shape == self.keras_model.layers[0].excite.bias.numpy().shape 
            assert self.params['dense2']['squeeze']['kernel'].shape == self.keras_model.layers[1].squeeze.kernel.numpy().shape 
            assert self.params['dense2']['squeeze']['bias'].shape == self.keras_model.layers[1].squeeze.bias.numpy().shape 
            assert self.params['dense2']['excite']['kernel'].shape == self.keras_model.layers[1].excite.kernel.numpy().shape 
            assert self.params['dense2']['excite']['bias'].shape == self.keras_model.layers[1].excite.bias.numpy().shape 
            assert self.params['dense3']['kernel'].shape == self.keras_model.layers[2].kernel.numpy().shape 
            assert self.params['dense3']['bias'].shape == self.keras_model.layers[2].bias.numpy().shape 

            assert np.allclose(self.params['dense1']['squeeze']['kernel'], self.keras_model.layers[0].squeeze.kernel.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense1']['squeeze']['bias'], self.keras_model.layers[0].squeeze.bias.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense1']['excite']['kernel'], self.keras_model.layers[0].excite.kernel.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense1']['excite']['bias'], self.keras_model.layers[0].excite.bias.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense2']['squeeze']['kernel'], self.keras_model.layers[1].squeeze.kernel.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense2']['squeeze']['bias'], self.keras_model.layers[1].squeeze.bias.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense2']['excite']['kernel'], self.keras_model.layers[1].excite.kernel.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense2']['excite']['bias'], self.keras_model.layers[1].excite.bias.numpy(), atol=self.atol)
            assert np.allclose(self.params['dense3']['kernel'], self.keras_model.layers[2].kernel.numpy(), atol=self.atol) 
            assert np.allclose(self.params['dense3']['bias'], self.keras_model.layers[2].bias.numpy(), atol=self.atol) 
            
            assert self.states['dense1']['bn1']['running_mean'].shape == self.keras_model.layers[0].bn1.moving_mean.numpy().shape 
            assert self.states['dense1']['bn1']['running_var'].shape == self.keras_model.layers[0].bn1.moving_variance.numpy().shape 
            assert self.states['dense1']['bn2']['running_mean'].shape == self.keras_model.layers[0].bn2.moving_mean.numpy().shape 
            assert self.states['dense1']['bn2']['running_var'].shape == self.keras_model.layers[0].bn2.moving_variance.numpy().shape 
            assert self.states['dense2']['bn1']['running_mean'].shape == self.keras_model.layers[1].bn1.moving_mean.numpy().shape 
            assert self.states['dense2']['bn1']['running_var'].shape == self.keras_model.layers[1].bn1.moving_variance.numpy().shape 
            assert self.states['dense2']['bn2']['running_mean'].shape == self.keras_model.layers[1].bn2.moving_mean.numpy().shape 
            assert self.states['dense2']['bn2']['running_var'].shape == self.keras_model.layers[1].bn2.moving_variance.numpy().shape 
            
            assert np.allclose(self.states['dense1']['bn1']['running_mean'], self.keras_model.layers[0].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn1']['running_var'], self.keras_model.layers[0].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_mean'], self.keras_model.layers[0].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense1']['bn2']['running_var'], self.keras_model.layers[0].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_mean'], self.keras_model.layers[1].bn1.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn1']['running_var'], self.keras_model.layers[1].bn1.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_mean'], self.keras_model.layers[1].bn2.moving_mean.numpy(), atol=self.atol) 
            assert np.allclose(self.states['dense2']['bn2']['running_var'], self.keras_model.layers[1].bn2.moving_variance.numpy(), atol=self.atol) 
            assert np.allclose(jaxmao_prediction, keras_prediction.numpy(), atol=self.atol)     

    def test_run_time(self):
        pass
    

  
if __name__ == '__main__':
    module = TestModule()
    module.test_single_dense()
    module.test_many_dense()
    module.test_nested_dense()
    
    nested_module = TestNestedDenseModules()
    nested_module.test_nested_dense_with_batch_normalization()
    nested_module.test_randomly_switch_trainable_on_nested_dense()
    nested_module.test_gradients()
    print('fisnih')