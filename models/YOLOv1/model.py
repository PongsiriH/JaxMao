import jax
from jaxmao.modules import Module, Conv2d, BatchNorm2d

leaky_relu = jax.
class ConvBnLRelu(Module):
    def __init__(self, in_channels, out_channels):
        self