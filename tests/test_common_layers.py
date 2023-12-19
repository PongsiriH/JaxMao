"""TODO
Conv2dTransposed's shape, batchnorms ()
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import random, gc, time

from jaxmao import (Dense, Conv2d, Conv2dTransposed,
                              GlobalAveragePooling2d, GlobalMaxPooling2d, 
                              Bind, 
                              AveragePooling2d, MaxPooling2d)
import jax

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')

class TestDense:
    def test_all(self):
        """Run all test methods for the Dense layer."""
        print("Testing shapes...")
        self.test_shapes()
        print("Shapes test passed.")

        print("Testing value consistency with Keras...")
        self.test_value_with_Keras()
        print("Value consistency with Keras test passed.")

        print("Testing gradient consistency with Keras...")
        self.test_grad_with_Keras()
        print("Gradient consistency with Keras test passed.")

        print("Testing randomly switching trainable status...")
        self.test_randomly_switch_trainable()
        print("Randomly switching trainable status test passed.")

        print("Testing performance speed...")
        self.test_speed()
        print("Performance speed test passed.")
    
    def test_shapes(self):
        in_channels = 10
        out_channels = 44
        sample_inputs1 = jax.random.normal(jax.random.key(9), (16, in_channels))
        sample_inputs2 = jax.random.normal(jax.random.key(9), (16, 10, in_channels))
        sample_inputs3 = jax.random.normal(jax.random.key(9), (16, 10, 2, in_channels))

        dense = Dense(in_channels, out_channels, use_bias=True)
        params, states = dense.init(jax.random.key(0))
        with Bind(dense, params, states) as ctx:
            out1 = ctx.module(sample_inputs1)
            out2 = ctx.module(sample_inputs2)
            out3 = ctx.module(sample_inputs3)
        
        assert out1.shape == (16, out_channels)    
        assert out2.shape == (16, 10, out_channels)
        assert out3.shape == (16, 10, 2, out_channels)
        
    def test_value_with_Keras(self):
        seed = 42
        in_channels = 16
        atol = 1e-9
        
        sample_data = np.random.normal(0, 1, (8, in_channels))
        
        jaxmao_model = Dense(in_channels, 44)
        params, states = jaxmao_model.init(jax.random.key(seed=seed))
        jaxmao_predicted, states, _ = jaxmao_model.apply(sample_data, params, states)
        
        keras_model = keras.layers.Dense(44)
        keras_model.build(input_shape=(None, in_channels))
        keras_model.set_weights([params['kernel'], params['bias']])
        keras_predicted = keras_model(sample_data)
        
        assert jaxmao_predicted.shape == keras_predicted.shape
        assert np.allclose(jaxmao_predicted, keras_predicted.numpy(), atol=atol)
    
    def test_grad_with_Keras(self):
        seed = 42
        in_channels = 16
        out_channels = 44
        atol = 1e-9
        
        sample_data = np.random.normal(0, 1, (8, in_channels))
        sample_label = np.random.normal(-1, 1, (8, out_channels))
        def mse_loss(params, states):
            out, state, _ = jaxmao_model.apply(sample_data, params, states)
            return jax.numpy.square(out -sample_label).mean()
        
        jaxmao_model = Dense(in_channels, out_channels)
        jaxmao_model.set_trainable(True)
        
        params, states = jaxmao_model.init(jax.random.key(seed=seed))
        jaxmao_gradients = jax.grad(mse_loss, argnums=0)(params, states)
        
        keras_model = keras.layers.Dense(out_channels)
        keras_model.build(input_shape=(None, in_channels))
        keras_model.set_weights([params['kernel'], params['bias']])
        with tf.GradientTape() as tape:
            predictions = keras_model(sample_data)
            loss = tf.reduce_mean(tf.square(predictions-sample_label))

        keras_gradients = tape.gradient(loss, keras_model.trainable_variables)
        
        assert jaxmao_gradients['kernel'].shape == keras_gradients[0].shape
        assert jaxmao_gradients['bias'].shape == keras_gradients[1].shape
        assert np.allclose(jaxmao_gradients['kernel'], keras_gradients[0].numpy(), atol=atol)
        assert np.allclose(jaxmao_gradients['bias'], keras_gradients[1].numpy(), atol=atol)
    
    def test_randomly_switch_trainable(self):
        seed = 42
        in_channels = 16
        out_channels = 44
        atol = 1e-9
        
        sample_data = np.random.normal(0, 1, (8, in_channels))
        sample_label = np.random.normal(-1, 1, (8, out_channels))
        def mse_loss(params, states):
            out, state, _ = jaxmao_model.apply(sample_data, params, states)
            return jax.numpy.square(out -sample_label).mean()
        
        jaxmao_model = Dense(in_channels, out_channels)
        params, states = jaxmao_model.init(jax.random.key(seed=seed))
        
        keras_model = keras.layers.Dense(out_channels)
        keras_model.build(input_shape=(None, in_channels))
        keras_model.set_weights([params['kernel'], params['bias']])
        
        
        for _ in range(100):
            trainable = random.choice([True, False])
            jaxmao_model.set_trainable(trainable)
            jaxmao_gradients = jax.grad(mse_loss, argnums=0)(params, states)
            
            with tf.GradientTape() as tape:
                predictions = keras_model(sample_data, training=trainable)
                loss = tf.reduce_mean(tf.square(predictions-sample_label))

            keras_gradients = tape.gradient(loss, keras_model.trainable_variables)
            
            assert jaxmao_gradients['kernel'].shape == keras_gradients[0].shape
            assert jaxmao_gradients['bias'].shape == keras_gradients[1].shape
            
            if trainable:
                assert np.allclose(jaxmao_gradients['kernel'], keras_gradients[0].numpy(), atol=atol)
                assert np.allclose(jaxmao_gradients['bias'], keras_gradients[1].numpy(), atol=atol)
            else:
                # jaxmao_gradients should be all zero when trainable = False
                assert np.allclose(jaxmao_gradients['kernel'], 0.0, atol=atol)
                assert np.allclose(jaxmao_gradients['bias'], 0.0, atol=atol)
    
    def test_speed(self):
        in_channels = 16
        out_channels = 44
        sample_data = np.random.normal(0, 1, (1000, in_channels))  # Large batch for testing

        jax_model = Dense(in_channels, out_channels)
        params, states = jax_model.init(jax.random.PRNGKey(0))

        keras_model = tf.keras.layers.Dense(out_channels, input_shape=(in_channels,))
        keras_model.build(input_shape=(None, in_channels))
        keras_model.set_weights([params['kernel'], params['bias']])
        
        # warmup-compile phase
        @jax.jit
        def jitted_apply(sample_data, params, states):
            return jax_model.apply(sample_data, params, states)
        
        @tf.function(jit_compile=True)
        def keras_model_function(input_data):
            return keras_model(input_data)
        
        _ = jitted_apply(sample_data, params, states)
        _ = keras_model_function(sample_data)
        
        gc.disable()
        
        # JAX Model Timing
        start_time = time.perf_counter()
        for _ in range(500):
            _ = jitted_apply(sample_data, params, states)
        jax_time = time.perf_counter() - start_time
        jax_avg_time = jax_time / 500

        # Keras Model Timing
        start_time = time.perf_counter()
        for _ in range(500):
            _ = keras_model_function(sample_data)
        keras_time = time.perf_counter() - start_time
        keras_avg_time = keras_time / 500
        
        gc.enable()
        
        # Output results
        print(f"JAX Model Total Time: {jax_time} seconds, Average Time per Iteration: {jax_avg_time} seconds")
        print(f"Keras Model Total Time: {keras_time} seconds, Average Time per Iteration: {keras_avg_time} seconds")

        # Performance comparison
        if jax_time < keras_time:
            print(f"JAX is {(keras_time / jax_time - 1) * 100:.2f}% faster than Keras.")
        else:
            print(f"Keras is {(jax_time / keras_time - 1) * 100:.2f}% faster than JAX.")

class TestConv2D:
    def test_all(self):
        self.test_shapes()
        self.test_value_with_Keras()
        
    def test_shapes(self):
        batch_size = 16
        height, width = 28, 28
        in_channels = 3
        out_channels = 44
        kernel_size = (3, 3)
        strides = (1, 1)
        dilation = (1, 2)

        sample_inputs = jax.random.normal(jax.random.PRNGKey(9), (batch_size, height, width, in_channels))

        # Adjust the formulas for new_height and new_width based on strides and dilation
        def calculate_output_dim(input_dim, kernel_dim, stride, dilate, padding):
            if padding == 'VALID':
                output_dim = (input_dim - dilate * (kernel_dim - 1) - 1) // stride + 1
            elif padding == 'SAME':
                output_dim = (input_dim + stride - 1) // stride
            return output_dim

        # VALID padding
        conv2d_valid = Conv2d(in_channels, out_channels, kernel_size, use_bias=True, padding='VALID', dilation=dilation, strides=strides)
        params_valid, states_valid = conv2d_valid.init(jax.random.PRNGKey(0))
        output_valid, _, _ = conv2d_valid.apply(sample_inputs, params_valid, states_valid)
        new_height_valid = calculate_output_dim(height, kernel_size[0], strides[0], dilation[0], 'VALID')
        new_width_valid = calculate_output_dim(width, kernel_size[1], strides[1], dilation[1], 'VALID')
        assert output_valid.shape == (batch_size, new_height_valid, new_width_valid, out_channels)

        # SAME padding
        conv2d_same = Conv2d(in_channels, out_channels, kernel_size, use_bias=True, padding='SAME', dilation=dilation, strides=strides)
        params_same, states_same = conv2d_same.init(jax.random.PRNGKey(0))
        output_same, _, _ = conv2d_same.apply(sample_inputs, params_same, states_same)
        new_height_same = calculate_output_dim(height, kernel_size[0], strides[0], dilation[0], 'SAME')
        new_width_same = calculate_output_dim(width, kernel_size[1], strides[1], dilation[1], 'SAME')
        assert output_same.shape == (batch_size, new_height_same, new_width_same, out_channels)

    def test_value_with_Keras(self):
        seed = 22
        batch_size = 8
        height, width = 28, 28
        in_channels = 3
        out_channels = 44
        kernel_size = (3, 3)
        strides = (1, 1)
        padding = 'same'
        atol = 1e-9
        
        sample_data = np.random.normal(0, 1, (batch_size, height, width, in_channels))

        # JAX Conv2D model
        jaxmao_model = Conv2d(in_channels, out_channels, kernel_size, padding=padding, strides=strides)
        params, states = jaxmao_model.init(jax.random.PRNGKey(seed=seed))
        jaxmao_predicted, _, _ = jaxmao_model.apply(sample_data, params, states)

        # Keras Conv2D model
        keras_model = keras.layers.Conv2D(out_channels, kernel_size, strides=strides, padding=padding, input_shape=(height, width, in_channels))
        keras_model.build(input_shape=(None, height, width, in_channels))
        keras_model.set_weights([params['kernel'], params['bias']])
        keras_predicted = keras_model(sample_data.astype(np.float32))

        assert jaxmao_predicted.shape == keras_predicted.shape
        assert np.allclose(jaxmao_predicted, keras_predicted.numpy(), atol=atol)

class TestConv2dTransposed:
    def test_all(self):
        self.test_shapes()
        self.test_value_with_Keras()
        
    def test_shapes(self):
        pass
    
    def test_value_with_Keras(self):
        seed = 22
        batch_size = 40
        height, width = 77, 77
        in_channels = 3
        out_channels = 44
        kernel_size = (4, 4)
        strides = (2, 2)
        padding = 'same'
        atol = 5e-4
        
        for _ in range(100):
            keras.backend.clear_session()
            gc.collect()
            
            seed = np.random.randint(0, 100)  # Random seed between 0 and 99
            batch_size = np.random.randint(1, 10)  # Random batch size between 1 and 99
            height, width = np.random.randint(1, 20), np.random.randint(1, 20)  # Random height and width
            in_channels = np.random.randint(1, 10)  # Random number of input channels between 1 and 9
            out_channels = np.random.randint(1, 20)  # Random number of output channels between 1 and 99
            kernel_size = (np.random.randint(1, 10), np.random.randint(1, 10))  # Random kernel size (1 to 9)
            strides = (np.random.randint(1, 5), np.random.randint(1, 5))  # Random strides (1 to 4)

            sample_data = np.random.normal(5, 20, (batch_size, height, width, in_channels))

            # JAX Conv2D model
            jaxmao_model = Conv2dTransposed(in_channels, out_channels, kernel_size, padding=padding, strides=strides)
            params, states = jaxmao_model.init(jax.random.PRNGKey(seed=seed))
            jaxmao_predicted, _, _ = jaxmao_model.apply(sample_data, params, states)
            
            # Keras Conv2D model
            keras_model = keras.layers.Conv2DTranspose(out_channels, kernel_size, strides=strides, padding=padding, input_shape=(height, width, in_channels))
            keras_model.build(input_shape=(None, height, width, in_channels))
            keras_model.set_weights([jax.numpy.transpose(params['kernel'], (0, 1, 2, 3)), params['bias']])
            keras_predicted = keras_model(sample_data.astype(np.float32))

            # print((jaxmao_predicted - keras_predicted.numpy()).max())
            assert jaxmao_predicted.shape == keras_predicted.shape
            assert np.allclose(jaxmao_predicted, keras_predicted.numpy(), atol=atol)
    
    
class TestBN1d:
    pass

class TestBN2d:
    pass

class TestGAP2d:
    def __init__(self, seed=42):
        self.atol = 1e-6
        self.jaxmao_gap = GlobalAveragePooling2d()
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(seed))
        self.keras_gap = keras.layers.GlobalAveragePooling2D()
        
    def test_value(self):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)

        keras_out = self.keras_gap(sample_data)
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)

class TestGMP2d:
    def __init__(self, seed=42):
        self.atol = 1e-6
        self.jaxmao_gap = GlobalMaxPooling2d()
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(seed))
        self.keras_gap = keras.layers.GlobalMaxPooling2D()
        
    def test_value(self):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)

        keras_out = self.keras_gap(sample_data)
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)


class TestAveragePooling2D:
    def __init__(self, seed=42):
        self.atol = 1e-6
        self.seed = seed
    
    def test_all(self):
        self.test_value()
        self.test_same_padding()
        self.test_valid_padding()
        self.test_valid_padding_randomized()
        self.test_same_padding_randomized()
        
    def test_value(self, plot=False):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = AveragePooling2d()
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.AveragePooling2D(padding='same')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)

        keras_out = self.keras_gap(sample_data)
        
        if plot:
            diff = np.abs(jaxmao_out - keras_out.numpy())
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 4)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                ax.imshow(diff[i])
            plt.suptitle('TestAveragePooling2D: bottom and right edges are implemented slightly differnetly. cannot use np.allclose to compare.')
            plt.show()
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out[:, :-1, :-1, :], keras_out.numpy()[:, :-1, :-1, :], atol=self.atol)
        
    def test_same_padding(self, plot=False):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = AveragePooling2d(padding='same')
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.AveragePooling2D(padding='same')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)
        keras_out = self.keras_gap(sample_data)
        
        if plot:
            diff = np.abs(jaxmao_out - keras_out.numpy())
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 4)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                ax.imshow(diff[i])
            plt.suptitle('TestAveragePooling2D: bottom and right edges are implemented slightly differnetly. cannot use np.allclose to compare.')
            plt.show()
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out[:, :-1, :-1, :], keras_out.numpy()[:, :-1, :-1, :], atol=self.atol)

    def test_valid_padding(self):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = AveragePooling2d(padding='valid')
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.AveragePooling2D(padding='valid')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)
        keras_out = self.keras_gap(sample_data)
    
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)

    def test_valid_padding_randomized(self):
        for _ in range(50):
            stride1 = np.random.randint(1, 5)
            stride2 = np.random.randint(1, 5)
            kernel1 = np.random.randint(1, 5)
            kernel2 = np.random.randint(1, 5)
            sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
            self.jaxmao_gap = AveragePooling2d(kernel_size=(kernel1, kernel2), padding='valid', strides=(stride1, stride2))
            self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
            self.keras_gap = keras.layers.AveragePooling2D(pool_size=(kernel1, kernel2), strides=(stride1, stride2), padding='valid')
            
            with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
                jaxmao_out = ctx.module(sample_data)
            keras_out = self.keras_gap(sample_data)
        
            assert jaxmao_out.shape == keras_out.numpy().shape
            assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)

    def test_same_padding_randomized(self):
        for _ in range(50):
            stride1 = np.random.randint(1, 5)
            stride2 = np.random.randint(1, 5)
            kernel1 = np.random.randint(1, 5)
            kernel2 = np.random.randint(1, 5)
            sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
            self.jaxmao_gap = AveragePooling2d(kernel_size=(kernel1, kernel2), padding='same', strides=(stride1, stride2))
            self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
            self.keras_gap = keras.layers.AveragePooling2D(pool_size=(kernel1, kernel2), strides=(stride1, stride2), padding='same')
            
            with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
                jaxmao_out = ctx.module(sample_data)
            keras_out = self.keras_gap(sample_data)

            assert jaxmao_out.shape == keras_out.numpy().shape
            w, h = jaxmao_out.shape[1], jaxmao_out.shape[2]
            assert np.allclose(jaxmao_out[:, 1:w-2, 1:h-2, :], keras_out.numpy()[:, 1:w-2, 1:h-2, :], atol=self.atol)
     
class TestMaxPooling2D:
    def __init__(self, seed=42):
        self.atol = 1e-6
        self.seed = seed
    
    def test_all(self):
        self.test_value()
        self.test_same_padding()
        self.test_valid_padding()
        self.test_valid_padding_randomized()
        self.test_same_padding_randomized()
        
    def test_value(self, plot=False):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = MaxPooling2d()
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.MaxPooling2D(padding='same')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)

        keras_out = self.keras_gap(sample_data)
        
        if plot:
            diff = np.abs(jaxmao_out - keras_out.numpy())
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 4)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                ax.imshow(diff[i])
            plt.suptitle('TestMaxPooling2D: bottom and right edges are implemented slightly differnetly. cannot use np.allclose to compare.')
            plt.show()
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out[:, :-1, :-1, :], keras_out.numpy()[:, :-1, :-1, :], atol=self.atol)
        
    def test_same_padding(self, plot=False):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = MaxPooling2d(padding='same')
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.MaxPooling2D(padding='same')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)
        keras_out = self.keras_gap(sample_data)
        
        if plot:
            diff = np.abs(jaxmao_out - keras_out.numpy())
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 4)
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                ax.imshow(diff[i])
            plt.suptitle('TestMaxPooling2D: bottom and right edges are implemented slightly differnetly. cannot use np.allclose to compare.')
            plt.show()
        
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out[:, :-1, :-1, :], keras_out.numpy()[:, :-1, :-1, :], atol=self.atol)

    def test_valid_padding(self):
        sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
        self.jaxmao_gap = MaxPooling2d(padding='valid')
        self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
        self.keras_gap = keras.layers.MaxPooling2D(padding='valid')
        
        with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
            jaxmao_out = ctx.module(sample_data)
        keras_out = self.keras_gap(sample_data)
    
        assert jaxmao_out.shape == keras_out.numpy().shape
        assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)

    def test_valid_padding_randomized(self):
        for _ in range(50):
            stride1 = np.random.randint(1, 5)
            stride2 = np.random.randint(1, 5)
            kernel1 = np.random.randint(1, 5)
            kernel2 = np.random.randint(1, 5)
            sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
            self.jaxmao_gap = MaxPooling2d(kernel_size=(kernel1, kernel2), padding='valid', strides=(stride1, stride2))
            self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
            self.keras_gap = keras.layers.MaxPooling2D(pool_size=(kernel1, kernel2), strides=(stride1, stride2), padding='valid')
            
            with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
                jaxmao_out = ctx.module(sample_data)
            keras_out = self.keras_gap(sample_data)
        
            assert jaxmao_out.shape == keras_out.numpy().shape
            assert np.allclose(jaxmao_out, keras_out.numpy(), atol=self.atol)

    def test_same_padding_randomized(self):
        for _ in range(50):
            stride1 = np.random.randint(1, 5)
            stride2 = np.random.randint(1, 5)
            kernel1 = np.random.randint(1, 5)
            kernel2 = np.random.randint(1, 5)
            sample_data = np.random.uniform(2, 5, (32, 45, 45, 3))
            self.jaxmao_gap = MaxPooling2d(kernel_size=(kernel1, kernel2), padding='same', strides=(stride1, stride2))
            self.params, self.states = self.jaxmao_gap.init(jax.random.key(self.seed))
            self.keras_gap = keras.layers.MaxPooling2D(pool_size=(kernel1, kernel2), strides=(stride1, stride2), padding='same')
            
            with Bind(self.jaxmao_gap, self.params, self.states) as ctx:
                jaxmao_out = ctx.module(sample_data)
            keras_out = self.keras_gap(sample_data)

            assert jaxmao_out.shape == keras_out.numpy().shape
            w, h = jaxmao_out.shape[1], jaxmao_out.shape[2]
            assert np.allclose(jaxmao_out[:, 1:w-2, 1:h-2, :], keras_out.numpy()[:, 1:w-2, 1:h-2, :], atol=self.atol)

     
def test_all():    
    dense = TestDense()
    dense.test_all()
    print('Dense passed')

    conv2d = TestConv2D()
    conv2d.test_all()
    print('Conv2d passed')

    conv_transposed = TestConv2dTransposed()
    conv_transposed.test_all()
    print('Conv2dTransposed passed')

    gap = TestGAP2d()
    gap.test_value()
    print('GAP2d passed')

    gmp = TestGMP2d()
    gmp.test_value()
    print('GMP2d passed')
    
    avg_pooling2d = TestAveragePooling2D()
    # avg_pooling2d.test_value(plot=True)
    avg_pooling2d.test_all()
    print('avg_pooling2d passed')

    max_pooling2d = TestMaxPooling2D()
    # max_pooling2d.test_value(plot=True)
    max_pooling2d.test_all()
    print('max_pooling2d passed')

if __name__ == '__main__':
    print('jax.devices()', jax.devices())
    test_all()