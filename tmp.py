import jax
import jax.numpy as jnp

# Define a simple data array and a binary mask
data = jnp.array([1.0, 2.0, 3.0, 4.0])
mask = jnp.array([1, 0, 1, 0])

# Function for mask * data
def f1(data):
    return jnp.sum(mask * data)

# Function for jnp.where(mask, data, 0)
def f2(data):
    return jnp.sum(jnp.where(mask, data, 0))

# Compute gradients
grad_f1 = jax.grad(f1)(data)
grad_f2 = jax.grad(f2)(data)

print(grad_f1, grad_f2)

