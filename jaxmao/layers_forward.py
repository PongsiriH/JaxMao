"dense"
def _forward_bias(self, params, x, state):
    z = jnp.dot(x, params['weights']) + params['biases']
    return self.activation(z), state

def _forward_no_bias(self, params, x, state):
    z = jnp.dot(x, params['weights'])
    return self.activation(z), state

"conv2d"
def forward_bias(self, params, x, state):
    x = lax.conv_general_dilated(x, params['weights'], 
                                    window_strides=self.strides,
                                    padding=self.padding,
                                    lhs_dilation=None,
                                    rhs_dilation=self.dilation,
                                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                    feature_group_count=self.feature_group_count
                                    ) 
    x = lax.add(x, params['biases'][None, None, None, :]) # (batch_size, width, height, out_channels)
    return self.activation(x), state

def forward_no_bias(self, params, x, state):
    x = lax.conv_general_dilated(x, params['weights'], 
                                    window_strides=self.strides,
                                    padding=self.padding,
                                    lhs_dilation=None,
                                    rhs_dilation=self.dilation,
                                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                    feature_group_count=self.feature_group_count
                                    ) 
    return self.activation(x), state

"batch norm"
def forward_train(self, params, x, state):
    batch_mean = jnp.mean(x, axis=self.axis_mean, keepdims=True)
    batch_var = jnp.var(x, axis=self.axis_mean, keepdims=True)
    new_running_mean = state['momentum'] * state['running_mean'] + (1 - state['momentum']) * batch_mean
    new_running_var = state['momentum'] * state['running_var'] + (1 - state['momentum']) * batch_var

    normalized_x = (x - batch_mean) / jnp.sqrt(batch_var + self.eps)
    scaled_x = normalized_x * params['gamma'] + params['beta']
    
    new_state = {
        'running_mean': new_running_mean,
        'running_var': new_running_var,
        'momentum' : state['momentum'],
    }
    return scaled_x, new_state

def forward_inference(self, params, x, state):
    normalized_x = (x - state['running_mean']) / jnp.sqrt(state['running_var'] + self.eps)
    scaled_x = normalized_x * params['gamma'] + params['beta']
    return scaled_x, state

"Dropout"
    def forward_train(self, params, x, state):
        keep_rate = 1 - state['drop_rate']
        key, subkey = jax.random.split(state['key'])
        mask = jax.random.bernoulli(subkey, keep_rate, x.shape)
        return x * mask / keep_rate, {'key': key, 'drop_rate': state['drop_rate'], 'training': state['training']}
        
    def forward_inference(self, params, x, state):
        return x, state

"Pooling2d"
    def _pool_forward(self, params, x, state):        
        return lax.reduce_window(
            x,
            init_value=self.init_value,
            computation=self.reducing_fn,
            window_dimensions=self.kernel_size,
            window_strides=self.strides,
            padding=self.padding_config
        ), state