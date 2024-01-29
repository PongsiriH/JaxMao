from backbone.model import *
import pickle
import jax
from jaxmao import Bind
import numpy as np

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    key = jax.random.key(42)
    clf, params, states = load_model('YOLOv3_2/backbone/results/003_backbone.pkl')
        
    params, states = clf.init_yolo_head(params, states, key)
    with Bind(clf, params, states) as ctx:
        predictions = ctx.module(np.random.normal(size=(4, 416, 416, 3)))

    [print('pred: ', p.shape) for p in predictions]