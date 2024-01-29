from model import YOLOBackboneResidual, yolo_head_config
import pickle

def load_model(path):
    with open(path, 'rb') as f:
        model, params, states = pickle.load(f)
    return model,params,states

# model, params, states = init_model(key)
model: YOLOBackboneResidual
# model, params, states = load_model('YOLOv3_3/backbone/results/002_best.pkl')
model, params, states = load_model('YOLOv3_3/backbone/results/yolov3_multiscales_best.pkl')
params, states = model.remove_clf_head(params, states)
model.config[1] = yolo_head_config
# model.num_classes = 80
# model._build_yolo_head()
model.save_model('YOLOv3_3/backbone/results/yolov3_multiscales_backbone.pkl', params, states)

# import jax
# from jaxmao import Bind
# import numpy as np
# key = jax.random.key(42)

# params, states = model.init_yolo_head(params, states, key)
# print(params.keys(), states.keys())
# print(params['yolo_head'].keys())
# print(model.yolo_head.submodules.keys())
# with Bind(model, params, states) as ctx:
#     predictions = ctx.module(np.random.normal(size=(4, 416, 416, 3)))

# [print('pred: ', p.shape) for p in predictions]
# [print('pred: ', jax.nn.sigmoid(p[..., 0])) for p in predictions]