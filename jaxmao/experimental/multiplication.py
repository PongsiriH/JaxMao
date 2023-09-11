import jax
import jax.numpy as jnp

A = jnp.array([
        [[9, 1, 6],
        [4, 9, 9],
        [9, 2, 3]]
    ], dtype=jnp.float32)

B = jnp.array([
        [9, 1, 6],
        [4, 9, 9],
        [9, 2, 3]
    ], dtype=jnp.float32)


print(A)

from jax import make_jaxpr
from jax import xla_computation
from jax import lax

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
f = lambda x, y : A @ A
f(A, A)

def h(x, y):
    return lax.dot_general(x, y, (((1,), (0,)), ((0,), ())))

def h(x, y):
    return lax.dot_general(x, y, ((2, 1), (0, 0)))

make_jaxpr(f)(A, A)

make_jaxpr(h)(A, B)
h(A, B)

A.shape, B.shape

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
lhs = jnp.array([ 
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0] 
])

rhs = jnp.array([ 
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0] 
])

g = lambda x, y : lax.dot_general(x, y, (((1,), (1,)), ((), ()) ))

g(lhs, rhs)

lhs.shape, rhs.shape
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

lhs = jnp.array([ 
    [ [1.0, 2.0],
    [3.0, 4.0] ],
    
    [ [5.0, 6.0],
    [7.0, 8.0] ] ])

rhs = jnp.array([ [ [1.0, 0.0],
[0.0, 1.0] ],
[ [1.0, 0.0],
[0.0, 1.0] ] ])

g = lambda x, y : lax.dot_general(x, y, (((2,), (1,)), ((0), (0)) ))

g(lhs, rhs)

lhs.shape, rhs.shape


import jax.numpy as jnp
from jax import lax

# Define your input batch x and current weights y
batch_size = 2
input_features = 3
output_features = 3

x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Example input batch (2 samples with 3 features each)
y = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Example current weights

# Expand dimensions to account for batch size
x_expanded = jnp.expand_dims(x, axis=1)  # Shape: (batch_size, 1, input_features)
y_expanded = jnp.expand_dims(y, axis=0)  # Shape: (1, output_features, input_features)

# Perform the fully connected layer operation with batching using jax.lax.dot_general
output = lax.dot_general(x_expanded, y_expanded, (((1,), (0,)), ((), ())))

# Remove the extra dimensions from the output
output = jnp.squeeze(output, axis=(1,))

print("Output:")
print(output)


batch_size = 16
in_channels = 5
out_channels = 7

key = jax.random.key(42)
x = jax.random.normal(key, (batch_size, 2, in_channels))
x.shape

W = jax.random.normal(key, (in_channels, out_channels))
W.shape

b = jax.random.normal(key, (out_channels, ))

x = x.reshape((batch_size*2, in_channels))
from jax import lax
xdim = len(x.shape)
mul = lax.dot_general(x, W,
                (((xdim-1), (0,)), ((), ()))
                )
mul.shape

sum = jnp.add(mul, b)
sum.shape

tmp = sum.copy()
tmp = tmp.reshape(batch_size*2, out_channels)

# Flatten
def flatten(x):
    return x.reshape(x.shape[0], -1).squeeze()
vflatten = jax.vmap(flatten, in_axes=(0))

def ravel(x):
    return x.ravel()
vravel = jax.vmap(ravel, in_axes=(0))
x = jax.random.normal(key, (batch_size, in_channels, ))
x.shape

flatten(x).shape
# vflatten(x).shape
ravel(x).shape
# vravel(x).shape

