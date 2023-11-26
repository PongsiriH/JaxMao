class DenseDense(Module):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(5, 3, use_bias=True)
        self.dense2 = Dense(3, 10, use_bias=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
# if __name__ == '__main__':

    
#     class DenseDenseThenDense(Module):
#         def __init__(self):
#             super().__init__()
#             self.dd2 = DenseDense()
#             self.dd1 = Dense(1, 2, use_bias=True)
    
#         def __call__(self, inputs):
#             return self.dd1(self.dd2(inputs))
    
#     dd = DenseDenseThenDense()
#     print(dd.dd1.params_.keys())
#     print(dd.dd1.param('weights'))
#     params = dd.init(jax.random.key(44))
#     print('\n\nparams', params)
#     print('dd1', dd.dd1.params_.get_value())
    
#     # print(params.keys())
#     print(_update_parameters(dd, params))
    
#     sample_inputs = jax.random.normal(jax.random.key(42), (7, 1))
#     out = dd(sample_inputs)
#     # print('dd1 after', _get_parameters(dd))
#     # sample_input = 
    
    
# if __name__ == '__main__':
#     dense = Dense(5, 10)
#     print('dense.params: ', dense.param('weights'))
#     params = dense.init(jax.random.key(42))
#     _update_parameters(dense, params)
#     sample_inputs = jax.random.normal(jax.random.key(0), (16, 5))
#     out = dense(sample_inputs)
#     print(out.shape)

if __name__ == '__main__':    
    dense = DenseDense()
    # print('dense.params: ', dense.param('weights'))
    params, states = dense.init(jax.random.key(42))
    # print(type(params))
    # print(states)
    # _update_parameters(dense, params)
    # print(dense.dense1.param('weights').shape)
    # print(dense.dense2.param('weights').shape)
    
    sample_inputs = jax.random.normal(jax.random.key(0), (16, 5))
    with jax.checking_leaks():
        out, states = dense.apply(sample_inputs, params, states)
    print(out.shape)
    print(states)