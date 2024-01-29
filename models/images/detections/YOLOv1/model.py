import sys
sys.path.append("/home/jaxmao/jaxmaov2_/YOLOInJaxMao/")
from backbone.model import YOLOBackbone, YOLO_backbone_config

import jax
from jaxmao import Module, Sequential, Conv2d, BatchNorm2d, Dense, BatchNorm1d, Dropout, GlobalAveragePooling2d
from jaxmao import initializers as init
from config import simplified_yolov1_arch
import pickle

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

class FullyConvolutionalYOLOv1(Module):
    def __init__(self, config=simplified_yolov1_arch, SBC=[7, 2, 4]):
        super().__init__()
        self.S, self.B, self.C = SBC
        self.in_channels = 3
        self.config = config
        self.conv = self._build_conv()
        self.fc = self._build_fc()
        
    def call(self, x: jax.Array):
        x = self.conv(x)
        x = self.fc(x).reshape(x.shape[0], self.S, self.S, self.B*5 + self.C)
        x = x.at[..., :3].set(2 * jax.nn.sigmoid(x[..., :3]) - 0.5)
        x = x.at[..., 5:8].set(2 * jax.nn.sigmoid(x[..., 5:8]) - 0.5)
        x = x.at[..., 10:].set(jax.nn.softmax(x[..., 10:]))
        return x
    
    def _build_conv(self):
        sequential = Sequential()
        in_channels = self.in_channels
        for layer in self.config:
            kernel_size, out_channels, strides = layer
            sequential.add(ConvBnLRelu(in_channels, out_channels, kernel_size, strides))
            in_channels = out_channels
        self.conv_out_channels = out_channels
        return sequential
        
    def _build_fc(self):
        sequential = Sequential([
            ConvBnLRelu(self.conv_out_channels, 1024, (1,1)),
            ConvBnLRelu(1024, 512, (1,1)),
            Conv2d(512, self.S*self.S*(self.B*5 + self.C), (1,1), kernel_init=init.GlorotUniform()),
            BatchNorm2d(self.S*self.S*(self.B*5 + self.C)),
            GlobalAveragePooling2d(),
        ])
        return sequential
    

class YOLOv1FromBackbone(Module):
    def __init__(self, path_backbone="/home/jaxmao/jaxmaov2_/YOLOInJaxMao/backbone/results/001_best.pkl", SBC=[7, 2, 4]):
        super().__init__()
        self.S, self.B, self.C = SBC
        self.backbone_out_channels = 512
        self.path_backbone = path_backbone
        self.config_ = [
            (1, 256, 1), (3, 512, 1),
            (1, 256, 1), (3, 512, 1),
        ]
        with open(self.path_backbone, 'rb') as f:
            self.bb, _, _ = pickle.load(f)
        self.bb: Module
        self.bb.submodules.pop('head')
        self.conv = self._build_conv()
        self.yolo_head = self._build_yolo_head()
    
    def init(self, key):
        params, states = super().init(key)
        with open(self.path_backbone, 'rb') as f:
            _, bb_params, bb_states = pickle.load(f)
        params['bb'] = {'backbone': bb_params['backbone']}
        states['bb'] = {'backbone': bb_states['backbone']}
        return params, states
    
    def call(self, x: jax.Array):
        x = self.bb.backbone(x)
        x = self.conv(x)
        x = self.yolo_head(x).reshape(x.shape[0], self.S, self.S, self.B*5 + self.C)
        x = x.at[..., :3].set(2 * jax.nn.sigmoid(x[..., :3]) - 0.5)
        x = x.at[..., 5:8].set(2 * jax.nn.sigmoid(x[..., 5:8]) - 0.5)
        x = x.at[..., 10:].set(jax.nn.softmax(x[..., 10:]))
        return x
    
    def _build_conv(self):
        sequential = Sequential()
        in_channels = self.backbone_out_channels
        for layer in self.config_:
            kernel_size, out_channels, strides = layer
            sequential.add(ConvBnLRelu(in_channels, out_channels, kernel_size, strides))
            in_channels = out_channels
        self.conv_out_channels = out_channels
        return sequential
        
    def _build_yolo_head(self):
        sequential = Sequential([
            ConvBnLRelu(self.conv_out_channels, 512, (1,1)),
            ConvBnLRelu(512, 256, (3,3)),
            Conv2d(256, self.S*self.S*(self.B*5 + self.C), (1,1), kernel_init=init.GlorotUniform()),
            BatchNorm2d(self.S*self.S*(self.B*5 + self.C)),
            GlobalAveragePooling2d(),
        ])
        return sequential

if __name__ == '__main__':
    key = jax.random.key(42)
    model = YOLOv1FromBackbone()
    params, states = model.init(key)
    
    images = jax.random.normal(key, (3, 45, 45, 3))
    predictions, states, _ = model.apply(images, params, states)
    print(predictions.shape)
    
# if __name__ == '__main__':
#     from jaxmao.modules import Summary
#     yolo = FullyConvolutionalYOLOv1(simplified_yolov1_arch)
#     print('yolo.conv', yolo.conv)
#     print('yolo.conv.submodules', yolo.conv.submodules)
#     print('yolo.submodules', yolo.submodules)
#     with Summary(yolo) as ctx:
#         ctx.summary((1, 224, 224, 3))
#         print(ctx.params_.keys())
#         print(ctx.params_['conv'].keys())
#         print(ctx.params_['fc'].keys())
#         print(jax.tree_util.tree_structure(ctx.params_))

