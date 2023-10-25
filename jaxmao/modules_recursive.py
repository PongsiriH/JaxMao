import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
os.environ['JAX_CHECK_TRACER_LEAKS'] = 'on'

from typing import Tuple
from utils_struct import (
                        RecursiveDict,
                        PostInitialization,
                        _check_dict_ids
                    )

from metrics import Metric
from jax import random, jit, value_and_grad

def _update_recursive_dict(lhs, rhs):
    for key in rhs.keys():
        lhs[key] = RecursiveDict(rhs[key])
    return lhs

def _copy_recursive_dict(rhs):
    lhs = rhs
    return _update_recursive_dict(lhs, rhs)


# class _ContextManager:
#     def __init__(self, model, new_params):
#         self.model = model
#         self.old_params = self.model.params

#         self.model.params = dict()
#         for key in model.params.keys():
#             self.model.params[key] = dict(model.params[key])
        
#         self.new_params = new_params
#         for key in new_params.keys():
#             self.new_params[key] = new_params[key]
#         # print('__init__: id new params ', _check_dict_ids(self.new_params, new_params))
#         # print('__init__: id model same ', id(self.model) == id(clf))
    
#     def __enter__(self):
#         # self.old_params = self.model.params
#         self.model.params = self.new_params
#         for key in self.new_params.keys():
#             self.model.params[key] = self.new_params[key]
#         # print('__enter__: id saved old params after', _check_dict_ids(self.old_params, self.model.params))
#         # print('__enter__: id model updated params ', _check_dict_ids(self.model.params, self.new_params))
#         return self

#     def __exit__(self, type, value, traceback):
#         print('__exit__ self.old_params', self.old_params)
#         self.model.params = self.old_params
#         # print('__exit__: id model old params ', _check_dict_ids(self.model.params, self.old_params))
#         # return self

class _ContextManager:
    def __init__(self, model, new_params):
        self.model = model
        self.old_params = model.params
        self.new_params = new_params
    
    def __enter__(self):
        self.model.params = RecursiveDict(self.new_params)

    def __exit__(self, type, value, traceback):
        self.model.params = self.old_params


    
class Module(metaclass=PostInitialization):
    is_collectable = True

    def __init__(self):
        self.layers = dict()
        self.state = RecursiveDict()
        self.params = RecursiveDict()
        
        self.num_params = 0
        self.num_states = 0

        self.training = False
        
        self.loss = None
        self.optimizer = None
        self.metrics = dict()
        
    def post_initialization(self):
        for (attr_name, obj) in self.__dict__.items():  # collect layers and params
            if hasattr(obj, 'is_collectable') and obj.is_collectable:
                self.layers[attr_name] = obj
                self.params[attr_name] = obj.params
                self.state[attr_name] = obj.state
                obj.name = attr_name
        self.num_layers = len(self.layers)

    def init_params(self, key):            
        for name in self.layers:
            key, subkey = random.split(key)
            self.layers[name].init_params(subkey)
            self.params[name] = self.layers[name].params 
        return self.params
    
    def update_params(self, new_params):
        for name in self.layers:
            self.params[name] = new_params[name]

    def update_state(self, new_state):
        for name in self.layers:
            self.state[name] = new_state[name]

    def __call__(self, x):
        raise NotImplementedError("The forward method should be overridden by subclass. Keep in mind that forward must return tuple(f(x), new_state) be pure (JAX-wise).")
    
    def _context(self, new_params):
        return _ContextManager(self, new_params)
    
    def apply(self, new_params, x):
        print('apply', self, self.name, self.layers)
        for name in self.layers.keys():
            x = self.layers[name].apply(new_params[name], x)
            
        with self._context(new_params):
            x = self.__call__(x)
            return x



if __name__ == '__main__':
    import numpy as np
    import jax.numpy as jnp
    import jax
    
    from utils import make_loss_function_gradable
    from losses import MeanSquaredError
    from optimizers import GradientDescent
    
    num_samples = 20
    x = np.linspace(0, 5, num_samples).reshape(num_samples, 1).astype('float32')
    y = (lambda x: 3*np.power(x, 2))(x).astype('float32')

    x = x / x.max()
    y = y / y.max()
    from layers import Dense, ReLU
    
    class Regressor(Module):
        def __init__(self):
            super().__init__()
            self.name = 'regressor'
            self.dense1 = Dense(1, 16, 'relu')
            self.dense2 = Dense(16, 1, 'relu')
        
        def __call__(self, x):
            x = self.dense2(self.dense1(x))
            return x

    reg = Regressor()
    reg.init_params(jax.random.key(44))

    print('len(reg.layers): ', len(reg.layers), reg.layers)
    print('len(reg.dense1.layers): ', len(reg.dense1.layers), reg.dense1.layers)
    print('len(reg.dense2.layers): ', len(reg.dense2.layers), reg.dense2.layers)
    
    reg2 = Regressor()
    reg2.init_params(jax.random.key(44))
    # another_params = {'dense1': {'activation': {}, 'weights': jnp.array([[55.5]], dtype='float32'), 'biases': jnp.array([1.0], dtype='float32')}}
    # print('__main__ reg.params    : ', reg.params)
    # print('__main__ another_params: ', another_params)
    
    print()
    print('out0', reg(x).ravel())
    
    print()
    out1 = reg.apply(reg.params, x=x)
    print('out1', out1.ravel())

    # print()
    # out2 = reg.apply(reg2.params, x=x)
    # print('out2', out2.ravel())


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, y, label='actual')
    # plt.plot(x, reg.apply(reg.params, x), label='pred')
    # plt.title('before train')
    # plt.legend()
    # plt.waitforbuttonpress()
    
    # EPOCHS = 10
    # loss_fn = MeanSquaredError()
    # optimizer = GradientDescent(lr=0.01, params=reg.params)
    
    # def _loss_fn(params, x, y):
    #     y_pred = reg.apply(params, x)
    #     loss = loss_fn(y_pred, y)
    #     return loss
    # loss_and_grad = jax.value_and_grad(_loss_fn, argnums=0, has_aux=False)
    
    # # print('reg.params', reg.params)
    # loss, gradients = loss_and_grad(reg.params, x, y)
    
    # for epoch in range(EPOCHS):
    #     loss, gradients = loss_and_grad(reg.params, x, y)
    #     new_params, optimizer.state = optimizer(reg.params, gradients, optimizer.state)
    #     for key in new_params:
    #         reg.params[key] = new_params[key]
    #     print(f'e{epoch}, loss: ', loss)
    #     print('gradients ', gradients['dense1']['biases'], gradients['dense2']['biases'])
        
    # # print('reg.params after training', reg.params)
    # y_pred = reg.apply(reg.params, x)
    # print('y_pred', y_pred.ravel())
    # plt.figure()
    # plt.plot(x, y, label='actual')
    # plt.plot(x, y_pred, label='pred')
    # plt.title('after train')
    # plt.legend()
    # plt.waitforbuttonpress()
    
    
    
    
# if __name__ == '__main__':
#     import jax
#     from sklearn.datasets import load_digits
#     from layers import Dense
#     from losses import CategoricalCrossEntropy
#     from optimizers import GradientDescent
    
#     images, targets = load_digits(return_X_y=True)

#     images = images / images.max()
#     targets_enc = jax.nn.one_hot(targets, num_classes=10)
    
#     class Classifier(Module):
#         def __init__(self):
#             super().__init__()
#             self.dense1 = Dense(64, 32, activation='relu', use_bias=True)
#             self.dense2 = Dense(32, 10, activation='softmax', use_bias=True)

#         def __call__(self, x):
#             x = self.dense1(x)
#             x = self.dense2(x)
#             return x
    
#     seed = 4
#     key = jax.random.key(seed)
#     key, key1, key2 = jax.random.split(key, 3)
    
#     clf = Classifier()
#     clf2 = Classifier()
#     params1 = clf.init_params(jax.random.PRNGKey(2)).copy()
#     params2 = clf2.init_params(jax.random.PRNGKey(421)).copy()
    
#     print('before id dense1', _check_dict_ids(clf.params['dense1'], clf.dense1.params))
#     print('before id dense2', _check_dict_ids(clf.params['dense2'], clf.dense2.params))
    
#     # print()
#     # print(_check_dict_ids(params1, params2))
#     # print(jax.tree_util.tree_map(lambda x, y: x - y, params1, params2))
#     """model.apply and _update_recursive_dict"""
#     # out1 = clf(images[:5])
#     # print(out1.shape, out1.sum(axis=1), out1) # good
    
#     # # print(clf.params)

    
#     # new_params = {
#     #     'dense1' : {
#     #         'activation' : {},
#     #         'weights' : jax.random.normal(jax.random.key(22), (64, 32)),
#     #     },
#     #     'dense2' : {
#     #         'activation' : {},
#     #         'weights' : jax.random.normal(jax.random.key(11), (32, 10)),
#     #     }}
#     # new_params2 = RecursiveDict()
#     # new_params2 = _update_recursive_dict(new_params2, new_params)
#     # print(jax.tree_util.tree_structure(clf.params))
#     # print(jax.tree_util.tree_structure(new_params2))
    
#     # new_params3 = _update_recursive_dict(clf.params, new_params2)
#     # print(jax.tree_util.tree_structure(new_params3))

#     # out2 = clf.apply(new_params3, images[:5])
#     # print(out2.shape, out2.sum(axis=1), out2)
    
    
#     """Train"""
#     loss_fn = CategoricalCrossEntropy(reduce_fn='mean_over_batch_size')
#     optimizer = GradientDescent(lr=0.01, params=clf.params)
        
#     def loss_fn_wrapped(model, new_params, x, y):
#         y_pred = model.apply(new_params, x)
#         loss = loss_fn(y_pred, y)
#         return loss
    
#     loss_and_grad = jax.value_and_grad(loss_fn_wrapped, argnums=1, has_aux=False)
    
    
#     EPOCHS = 5
#     BATCH_SIZE = 128    
#     NUM_BATCHES = len(images) // BATCH_SIZE
#     for epoch in range(1):
#         total_losses = 0.0
#         for n in range(2): 
#             # loss, gradients = loss_and_grad(clf, new_parms_train, images, targets_enc)
#             # new_params, optimizer.state = optimizer.step(clf.params, gradients, optimizer.state)
#             # print('dense1 biases graident: ', gradients['dense1']['biases'])      
#             # clf.update_params(new_params)
#             with clf._context(params1) as manager:
#                 tout1 = manager.model(images[:10])
#             with clf2._context(params1) as manager:
#                 tout2 = manager.model(images[:10])
#             print('__main__: tout1 - tout2', tout1 - tout2)
#             # print('apply new_params diff: ', clf.apply(new_parms_train, images[:10]) - clf.apply(clf.params, images[:10]))

#             # total_losses += loss
#         print('epoch: {} - avg_loss: {} '.format(epoch+1, total_losses/NUM_BATCHES))
    
    
#     from sklearn.metrics import accuracy_score, precision_score, recall_score
#     y_pred = clf(images).argmax(axis=1)
#     accuracy = accuracy_score(targets, y_pred)
#     precision = precision_score(targets, y_pred, average='macro')
#     recall = recall_score(targets, y_pred, average='macro')
#     print('Accuracy : {:<.6f}'.format(accuracy))
#     print('Precision: {:<.6f}'.format(precision))
#     print('Recall   : {:<.6f}'.format(recall))

#     print(clf.params)
#     print('after id dense1', _check_dict_ids(clf.params['dense1'], clf.dense1.params))
#     print('after id dense2', _check_dict_ids(clf.params['dense2'], clf.dense2.params))
