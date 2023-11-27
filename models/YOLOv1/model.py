import jax
from jaxmao.modules import Module, Sequential, Conv2d, BatchNorm2d, Dense, BatchNorm1d, Dropout
import jaxmao.initializers as init
from config import simplified_yolov1_arch

class ConvBnLRelu(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), strides=(1,1), lrelu_constant=0.1):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, strides=strides, padding='SAME', 
                           kernel_init=init.HeNormal())
        self.bn = BatchNorm2d(out_channels)
        self.alpha = lrelu_constant
    
    def call(self, x):
        return jax.nn.leaky_relu(self.bn(self.conv(x)), self.alpha)

class DenseBnLRelu(Module):
    def __init__(self, in_channels, out_channels, lrelu_constant=0.1):
        super().__init__()
        self.dense = Dense(in_channels, out_channels,
                           kernel_init=init.HeNormal())
        self.bn = BatchNorm1d(out_channels)
        self.alpha = lrelu_constant
    
    def call(self, x):
        return jax.nn.leaky_relu(self.bn(self.dense(x)), self.alpha)
    
class YOLOv1(Module):
    def __init__(self, config=simplified_yolov1_arch, SBC=[7, 2, 4]):
        super().__init__()
        self.S, self.B, self.C = SBC
        self.in_channels = 3
        self.config = config
        self.conv = self._build_conv()
        self.fc = self._build_fc()
        
    def call(self, x: jax.Array):
        x = self.conv(x)
        # print('x.shape', x.shape)
        x = x.reshape(x.shape[0], -1)
        # print('x.shape', x.shape)
        x = self.fc(x).reshape(x.shape[0], self.S, self.S, self.B*5 + self.C)
        x[..., :3] = jax.nn.sigmoid(x[..., :3])
        x[..., 5:8] = jax.nn.sigmoid(x[..., 5:8])
        x[..., 10:] = jax.nn.sigmoid(x[..., 10:])
        # print('x.shape', x.shape)
        return x
    
    def _build_conv(self):
        sequential = Sequential()
        in_channels = self.in_channels
        for layer in self.config:
            kernel_size, out_channels, strides = layer
            sequential.add(ConvBnLRelu(in_channels, out_channels, kernel_size, strides))
            in_channels = out_channels
        return sequential
        
    def _build_fc(self):
        sequential = Sequential([
            DenseBnLRelu(3136, 512),
            DenseBnLRelu(512, 256),
            Dense(256, self.S*self.S*(self.B*5 + self.C), kernel_init=init.GlorotUniform()),
            BatchNorm1d(self.S*self.S*(self.B*5 + self.C))
        ])
        return sequential
    
if __name__ == '__main__':
    from jaxmao.modules import Summary
    yolo = YOLOv1(simplified_yolov1_arch)
    print('yolo.conv', yolo.conv)
    print('yolo.conv.submodules', yolo.conv.submodules)
    print('yolo.submodules', yolo.submodules)
    with Summary(yolo) as ctx:
        ctx.summary((1, 224, 224, 3))
        print(ctx.params_.keys())
        print(ctx.params_['conv'].keys())
        print(ctx.params_['fc'].keys())
        print(jax.tree_util.tree_structure(ctx.params_))

