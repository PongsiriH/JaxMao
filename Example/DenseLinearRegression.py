num_samples = 20
x = np.linspace(0, 1, num_samples).reshape(num_samples, 1).astype('float32')
x = x / x.max()
y = (lambda x: 3*x)(x).astype('float32')


"""Method 1
dense = Dense(1, 1, use_bias=False)
"""

"""Method 2
class LinearRegressor(Module):
    def __init__(self):
        super().__init__()
        self.dense = Dense(1, 1, use_bias=False)
    
    def __call__(self, x):
        return self.dense(x)

dense = LinearRegressor()
"""

dense.init_params(jax.random.key(0))

EPOCHS = 60
loss_fn = MeanSquaredError()
optimizer = GradientDescent(lr=0.1, params=dense.params)

def _loss_fn(params, x, y):
    y_pred = dense.apply(params, x)
    loss = loss_fn(y_pred, y)
    return loss
loss_and_grad = jax.value_and_grad(_loss_fn, argnums=0, has_aux=False)
loss, gradients = loss_and_grad(dense.params, x, y)

new_params = tree_map(lambda A: A+200, dense.params)
for epoch in range(EPOCHS):
    loss, gradients = loss_and_grad(new_params, x, y)
    new_params, optimizer.state = optimizer(dense.params, gradients, optimizer.state)
    dense.update_params(new_params)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, y, label='actual')
plt.plot(x, dense.apply(dense.params, x), label='apply pred')
plt.title('after train')
plt.legend()
plt.waitforbuttonpress()