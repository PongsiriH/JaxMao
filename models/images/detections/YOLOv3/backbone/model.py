from sys import implementation
import jax
import jax.numpy as jnp
from numpy import isin
from jaxmao import Module, Sequential, Conv2d, BatchNorm2d, Dense, BatchNorm1d, Dropout, GlobalAveragePooling2d
from jaxmao import initializers as init
from jaxmao import losses
import pickle
from typing import Tuple, Dict

# backbone_config = [
#     (16, 3, 1),
#     (32, 3, 2),
#     ["B", 2],
#     (64, 3, 2),
#     ["B", 4],
#     (128, 3, 2),
#     ["B", 8],
#     (256, 3, 2),
#     ["B", 8],
#     (512, 3, 2),
#     ["B", 4]
# ]

# yolo_head_config = [
#     (256, 1, 1),
#     (512, 3, 1),
#     "S",
#     (128, 1, 1),
#     "U",
#     (64, 1, 1),
#     (256, 3, 1),
#     "S",
#     (64, 1, 1),
#     "U",
#     (64, 1, 1),
#     (256, 3, 1),
#     "S",
# ]

### fat-backbone
# backbone_config = [
#     (32, 3, 1),
#     (64, 3, 2),
#     ["B", 8],
#     (128, 3, 2),
#     ["B", 8],
#     (256, 3, 2),
#     ["B", 4],
# ]

# yolo_head_config = [
#     (128, 1, 1),
#     (256, 3, 1),
#     "S",
#     (64, 1, 1),
#     "U",
#     (64, 1, 1),
#     (256, 3, 1),
#     "S",
#     (32, 1, 1),
#     "U",
#     (64, 1, 1),
#     (256, 3, 1),
#     "S",
# ]

backbone_config = [
    (32, 3, 1),
    (32, 3, 2),
    ["B", 1],
    (64, 3, 2),
    ["B", 2],
    (128, 3, 2),
    ["B", 8],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 4],
]

yolo_head_config = [
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
    (64, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

# yolo_config = backbone_config + yolo_head_config

class Sigmoid(Module):
    def call(self, x):
        return jax.nn.sigmoid(x)

class Softmax(Module):
    def call(self, x):
        return jax.nn.softmax(x)

class ConvBnLRelu(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), strides=(1,1), lrelu_constant=0.1, name=None):
        super().__init__(name=name)
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, strides=strides, padding='SAME', 
                           kernel_init=init.GlorotUniform())
        self.bn = BatchNorm2d(out_channels)
        self.alpha = lrelu_constant
    
    def call(self, x):
        return jax.nn.leaky_relu(self.bn(self.conv(x)), self.alpha)
    
class ResidualBlocks(Module):
    def __init__(self, channels, use_residual=True, num_repeats=1, concat=False, name=None):
        super().__init__(name=name)
        self.layers = Sequential([
            Sequential([ConvBnLRelu(channels, channels//2, kernel_size=1), ConvBnLRelu(channels//2, channels, kernel_size=3)]) 
            for repeat in range(num_repeats)])
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        self.concat = concat
            
    def call(self, x):
        for layer in self.layers.submodules.values():
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x
    
class ScalePrediction(Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.C = num_classes
        out_channels = 3*(self.C+5) # 3 anchors
        self.pred = Sequential([
            Conv2d(in_channels, out_channels, kernel_size=1, kernel_init=init.GlorotUniform()),
        ])
        
    def call(self, x):
        # x : [bs, w, h, c*3] -> [bs, 3, w, h, c]
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, x.shape[1], x.shape[2], self.C+5)
        )  

class Upsample(Module):
    def __init__(self, factor, method='nearest'):
        super().__init__()
        self.factor = factor
        self.method = method

    def call(self, x):
        out_shape = x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor, x.shape[3]
        return jax.image.resize(x, out_shape, method=self.method)

class UpsampleConv(Module):
    def __init__(self, factor, in_channels, method='nearest'):
        super().__init__()
        self.factor = factor
        self.method = method
        self.conv1 = Conv2d(in_channels, in_channels//2, (1,1), kernel_init=init.GlorotUniform())
        self.conv2 = Conv2d(in_channels//2, in_channels, (1,1), kernel_init=init.GlorotUniform())

    def call(self, x):
        out_shape = x.shape[0], x.shape[1]*self.factor, x.shape[2]*self.factor, x.shape[3]
        return self.conv2(jax.image.resize(self.conv1(x), out_shape, method=self.method))
    
class YOLOBackboneResidual(Module):
    def __init__(self, config=[backbone_config, yolo_head_config], in_channels=3, num_classes=100, clf_mode=True):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.config = config
        self.backbone = Sequential(self._build_conv(config[0]))
        self.clf_head = None
        self.yolo_head = None
        if clf_mode:
            self.clf_head = self._build_clf_head()
        
    def call(self, x: jax.Array):
        route_connections = []

        for layer in self.backbone.submodules.values():
            x = layer(x)
            if isinstance(layer, ResidualBlocks) and layer.num_repeats == 8:
                route_connections.append(x)
            
        if self.yolo_head:
            outputs = []
            for layer in self.yolo_head.submodules.values():
                if isinstance(layer, ScalePrediction):
                    outputs.append(layer(x))
                    continue
                
                x = layer(x)
                
                if isinstance(layer, Upsample):
                    x = jnp.concatenate([x, route_connections[-1]], axis=-1)
                    route_connections.pop()
                    
            return outputs
        
        elif self.clf_head:
            x = self.clf_head(x)
            
        return x
    
    def remove_clf_head(self, params: dict, states: dict) -> (dict, dict):
        params.pop('clf_head', None)
        states.pop('clf_head', None)
        self.submodules.pop('clf_head', None)
        return params, states
    
    def _build_yolo_head(self):
        layers = self._build_conv(self.config[1])
        self.yolo_head = Sequential(layers)
        self.submodules['yolo_head'] = self.yolo_head
        
    def init_yolo_head(self, params: dict, states: dict, key):
        self.clf_head = None
        
        h_params, h_states = self.yolo_head.init(key)
        params['yolo_head'] = h_params # add params for yolo head
        states['yolo_head'] = h_states
        return params, states
    
    def replace_scale_prediction(self, params, states, key=None, in_channels=[512, 256, 256]):
        if key is None: key = jax.random.key(42)
        self.yolo_head: Sequential
        for name in self.yolo_head.submodules:
            if isinstance(self.yolo_head.submodules[name], ScalePrediction):
                self.yolo_head.submodules[name] = ScalePrediction(in_channels[0], self.num_classes)
                params['yolo_head'][name], states['yolo_head'][name] = self.yolo_head.submodules[name].init(key)
                in_channels = in_channels[1:]
        return params, states
    
    def save_model(self, path, params, states):
        with open(path, 'wb') as f:
            pickle.dump((self, params, states), f)
    
    def _build_conv(self, config):
        sequential = []
        in_channels = self.in_channels
        for layer in config:
            print('config layer: ', layer)
            if isinstance(layer, tuple):
                out_channels, kernel_size, strides = layer
                sequential.append(ConvBnLRelu(in_channels, out_channels, kernel_size, strides))
                in_channels = out_channels
            elif isinstance(layer, list):
                if layer[0] == "B":
                    concat = layer[1] == 8
                    use_residual = True
                    sequential.append(ResidualBlocks(in_channels, use_residual, layer[1], concat))
            elif isinstance(layer, str):
                if layer == "S":
                    sequential.append(ResidualBlocks(in_channels, use_residual=False, num_repeats=1, concat=False))
                    sequential.append(ConvBnLRelu(in_channels, in_channels // 2, kernel_size=1))
                    sequential.append(ScalePrediction(in_channels // 2, num_classes=self.num_classes))
                    in_channels = in_channels // 2          
                elif layer == "U":
                    sequential.append(Upsample(factor=2))
                    in_channels = in_channels * 3
                elif layer == "C":
                    sequential.append(Upsample(factor=1))
                    in_channels = in_channels * 3

        self.backbone_out_channels = in_channels
        self.in_channels = in_channels
        return sequential
        
    def _build_clf_head(self):
        sequential = Sequential([
            Conv2d(self.in_channels, self.num_classes, (1,1), kernel_init=init.GlorotUniform()),
            BatchNorm2d(self.num_classes),
            GlobalAveragePooling2d()
        ])
        return sequential
    
if __name__ == '__main__':
    clf: YOLOBackboneResidual
    params: dict
    states: dict
    with open('YOLOv3_3/backbone/results/001_best.pkl', 'rb') as f:
        clf, params, states = pickle.load(f)
    # clf.save_backbone('YOLOv3_3/backbone/results/001_base.pkl', params, states)
    params, states = clf.remove_clf_head(params, states)
    clf.config[1] = yolo_head_config
    clf._build_yolo_head()
    clf.save_model('YOLOv3_3/backbone/results/001_base.pkl', params, states)
    
    from jaxmao import Bind
    import numpy as np
    key = jax.random.key(42)
    params, states = clf.init_yolo_head(params, states, key)
    print(params.keys(), states.keys())
    print(params['yolo_head'].keys())
    print(clf.yolo_head.submodules.keys())
    with Bind(clf, params, states) as ctx:
        predictions = ctx.module(np.random.normal(size=(4, 416, 416, 3)))
    
    [print('pred: ', p.shape) for p in predictions]