# JaxMao: Another Machine learning library using Jax.

If you find this, you are probably in a similar learning journey as me. I hope for this resource to be helpful.

I have build this library as part of my personal learning project. My goal is to build various machine learning algorithms using the JAX library.

# Classifier
```python
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
