class TestConv2dTransposed:
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
        dilation = (1, 1)

        sample_inputs = jax.random.normal(jax.random.PRNGKey(9), (batch_size, height, width, in_channels))

        # Adjust the formulas for new_height and new_width based on strides and dilation
        def calculate_output_dim(input_dim, kernel_dim, stride, dilate, padding):
            if padding == 'VALID':
                output_dim = (input_dim - dilate * (kernel_dim - 1) - 1) // stride + 1
            elif padding == 'SAME':
                output_dim = (input_dim + stride - 1) // stride
            return output_dim

        # VALID padding
        conv2d_valid = Conv2dTransposed(in_channels, out_channels, kernel_size, use_bias=True, padding='VALID', dilation=dilation, strides=strides)
        params_valid, states_valid = conv2d_valid.init(jax.random.PRNGKey(0))
        output_valid, _, _ = conv2d_valid.apply(sample_inputs, params_valid, states_valid)
        new_height_valid = calculate_output_dim(height, kernel_size[0], strides[0], dilation[0], 'VALID')
        new_width_valid = calculate_output_dim(width, kernel_size[1], strides[1], dilation[1], 'VALID')
        assert output_valid.shape == (batch_size, new_height_valid, new_width_valid, out_channels)

        # SAME padding
        conv2d_same = Conv2dTransposed(in_channels, out_channels, kernel_size, use_bias=True, padding='SAME', dilation=dilation, strides=strides)
        params_same, states_same = conv2d_same.init(jax.random.PRNGKey(0))
        output_same, _, _ = conv2d_same.apply(sample_inputs, params_same, states_same)
        new_height_same = calculate_output_dim(height, kernel_size[0], strides[0], dilation[0], 'SAME')
        new_width_same = calculate_output_dim(width, kernel_size[1], strides[1], dilation[1], 'SAME')
        assert output_same.shape == (batch_size, new_height_same, new_width_same, out_channels)

    def test_value_with_Keras(self):
        seed = 22
        batch_size = 40
        height, width = 77, 77
        in_channels = 3
        out_channels = 44
        kernel_size = (2, 2)
        strides = (1, 1)
        padding = 'same'
        atol = 1e-6
        
        sample_data = np.random.normal(0, 1, (batch_size, height, width, in_channels))

        # JAX Conv2D model
        jaxmao_model = Conv2dTransposed(in_channels, out_channels, kernel_size, padding=padding, strides=strides)
        params, states = jaxmao_model.init(jax.random.PRNGKey(seed=seed))
        jaxmao_predicted, _, _ = jaxmao_model.apply(sample_data, params, states)
        # print('kernel', params['kernel'].shape)
        # print('kernel', jax.numpy.transpose(params['kernel'], (1, 0, 2, 3)).shape)
        
        # Keras Conv2D model
        keras_model = keras.layers.Conv2DTranspose(out_channels, kernel_size, strides=strides, padding=padding, input_shape=(height, width, in_channels))
        keras_model.build(input_shape=(None, height, width, in_channels))
        print('\njaxmao', params['kernel'].shape, params['bias'].shape, '\n\n')
        print('\nkeras', keras_model.weights[0].shape, keras_model.weights[1].shape, '\n\n')
        keras_model.set_weights([jax.numpy.transpose(params['kernel'], (1, 0, 3, 2)), params['bias']])
        keras_predicted = keras_model(sample_data.astype(np.float32))

        print(jaxmao_predicted.shape, keras_predicted.shape)
        assert jaxmao_predicted.shape == keras_predicted.shape
        print((jaxmao_predicted - keras_predicted.numpy()).max())
        assert np.allclose(jaxmao_predicted, keras_predicted.numpy(), atol=atol)

if __name__ == '__main__':
    c = TestConv2D()
    c.test_all()
    print('CONV PASS')
    
    # ct = TestConv2dTransposed()
    # ct.test_value_with_Keras()