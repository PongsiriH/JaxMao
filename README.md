# JaxMao: Another Machine learning library using Jax.

Mostly Neural Network and Conformal predictions for now. I will expand in a future.

<hr>

# Classifier
```python
from jaxmao import Module, Dense
from jaxmao import regularizers
import jax

class MnistClassifier(Module):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64, 48, kernel_reg=regularizers.L1(0.001))
        self.dense2 = Dense(48, 32, kernel_reg=regularizers.L1(0.001))
        self.dense3 = Dense(32, 10)
        
    def call(self, x):
        x = jax.nn.relu(self.dense1(x))
        x = jax.nn.relu(self.dense2(x))
        x = jax.nn.softmax(self.dense3(x))
        return x
```