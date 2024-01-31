from model import *

class YOLOv3Model(Module):
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
        route_connections = list()

        for layer in self.backbone.submodules.values():
            layer_type = str(type(layer))
            x = layer(x)
            if (isinstance(layer, ResidualBlocks) or "ResidualBlocks" in layer_type) and layer.num_repeats == 8:
                route_connections.append(x)
            
        if self.yolo_head:
            outputs = []
            for layer in self.yolo_head.submodules.values():
                layer_type = str(type(layer))
                if isinstance(layer, ScalePrediction) or "ScalePrediction" in layer_type:
                    outputs.append(layer(x))
                    continue
                
                x = layer(x)
                
                if isinstance(layer, Upsample) or "Upsample" in layer_type:
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
            if isinstance(self.yolo_head.submodules[name], ScalePredictionv2):
                self.yolo_head.submodules[name] = ScalePredictionv2(in_channels[0], self.num_classes)
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
                    sequential.append(ScalePredictionv2(in_channels // 2, num_classes=self.num_classes))
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