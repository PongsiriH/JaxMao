from jaxmao import Bind
import numpy as np
import pickle
import jax
key = jax.random.key(42)

with open('YOLOv3.2/backbone/results/001_base.pkl', 'rb') as f:
    clf, params, states = pickle.load(f)
        
params, states = clf.init_yolo_head(params, states, key)
print(params.keys(), states.keys())
print(params['yolo_head'].keys())
print(clf.yolo_head.submodules.keys())
with Bind(clf, params, states) as ctx:
    predictions = ctx.module(np.random.normal(size=(4, 416, 416, 3)))

[print('pred: ', p.shape) for p in predictions]