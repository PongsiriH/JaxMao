from .utils_struct import (
                        FrozenDict,
                        PostInitialization,
                    )
from contextlib import ExitStack
import jax

class _ContextManager:
    def __init__(self, model, new_params, new_state, layers):
        self.model = model
        self.old_params = FrozenDict(model.params)
        self.new_params = FrozenDict(new_params)
        self.old_state = FrozenDict(model.state)
        self.new_state = FrozenDict(new_state)
        self.layers = layers
        
    def __enter__(self):
        self.model.params = FrozenDict(self.new_params)
        self.model.state = FrozenDict(self.new_state)            
        
        # for name in self.new_params: # !!! THIS FOR-LOOP CAUSE LEAKAGE
        #     with self.model.layers[name]._context(self.new_params[name]):
        #         self.model.layers[name].params = FrozenDict(self.new_params[name])
            
    def __exit__(self, type, value, traceback):
        self.model.params = FrozenDict(self.old_params)
        self.model.state = FrozenDict(self.old_state)
        self.new_params = None
        self.new_state = None
        self.old_params = None
        self.old_state = None
    
class RecursiveLayerContextManager:
    def __init__(self, layer, new_params):
        self.layer = layer
        self.old_params = layer.params
        self.new_params = new_params

    def __enter__(self):
        self.layer.params = self.new_params
        # If the layer has sub-layers, apply the context manager recursively
        if hasattr(self.layer, 'layers'):
            self.old_layers_params = {name: layers.params for name, layers in self.layer.layers.items()}
            for name, sub_layer in self.layer.layers.items():
                with RecursiveLayerContextManager(sub_layer, self.new_params.get(name, {})):
                    pass

    def __exit__(self, type, value, traceback):
        # If the layer has sub-layers, restore their original params
        if hasattr(self.layer, 'layers'):
            for name, sub_layer in self.layer.layers.items():
                sub_layer.params = self.old_sub_layer_params.get(name)
        # Restore the original params for this layer
        self.layer.params = self.old_params
           
# # @jax.jit
# def apply(model, layers, params, x):
#     with model._context(params, model.state):
#         x = model.__call__(x)
#     return x
    
class Module(metaclass=PostInitialization):
    is_collectable = True

    def __init__(self, name='main_module'):
        self.name = name
        self.num_params = 0
        self.num_states = 0
        
        self.training = False
        self.layers = dict()
        self.state = dict()
        self.params = dict()
        
    def post_initialization(self):            
        for (attr_name, obj) in self.__dict__.items():  # collect layers and params
            if hasattr(obj, 'is_collectable') and obj.is_collectable:
                self.layers[attr_name] = obj
                self.params[attr_name] = obj.params
                self.state[attr_name] = obj.state
                obj.name = attr_name
        self.num_layers = len(self.layers)

    # def post_initialization(self):
    #     for (attr_name, obj) in self.__dict__.items():  # collect layers and params
    #         if hasattr(obj, 'is_collectable') and obj.is_collectable:
    #             self.layers[attr_name] = obj
    #             self.params[attr_name] = obj.params
    #             self.state[attr_name] = obj.state
    #             obj.name = attr_name
                
    #             # Set the parent of the obj to be the current object (self)
    #             obj.parent = self
                
    #     self.params = FrozenDict(self.params)
    #     self.num_layers = len(self.layers)
    
    # def update_root_from_leaf(self, obj, attr_name=None):
    #     if attr_name:
    #         self.layers[attr_name] = obj
    #         self.params[attr_name] = obj.params
    #         self.state[attr_name] = obj.state
    #         obj.name = attr_name

    #     if hasattr(self, 'parent') and self.parent is not None:
    #         self.parent.update_root_from_leaf(self, self.name)


    def update_params(self, new_params):
        self.params = new_params
        
    def init_params(self, key):
        for name in self.layers:
            key, subkey = jax.random.split(key)
            self.layers[name].init_params(subkey)
            self.params[name] = self.layers[name].params
        self.params = FrozenDict(self.params)
        
    def update_state(self):
        for child in self.layers.values():
            child.update_state()
        
    def __call__(self, x):
        raise NotImplementedError("The __call__ method should be overridden by subclass. Keep in mind that forward must return tuple(f(x), new_state) be pure (JAX-wise).")

    def apply(self, new_params, x):
        with self._context(new_params), ExitStack() as stack:
            for name in new_params:
                stack.enter_context(RecursiveLayerContextManager(self.layers[name], new_params[name]))
            x = self.__call__(x)
        return x
            
    def _context(self, new_params):
        # self._recursive_update_params(self, new_params)
        return _ContextManager(self, new_params, self.state, self.layers)
    
def apply(module, params, *args, **kwargs):
    for name, values in params.items():
        setattr(module, 'params', values)
    return module(*args, **kwargs)

    # def propagate_params(self, new_params, childs):
    #     self.params = jax.tree_util.tree_map(lambda x, y: y, self.params, new_params)
    #     for name, child in childs.items():
    #         if isinstance(child, Module):
    #             child_params = new_params.get(name, {})
    #             setattr(child, "params", jax.tree_map(lambda x, y: y, child.params, child_params))
                
    #             # setattr(child, "params", new_params.get(name, {}))
    #             # child.params = new_params[name]
    #             # child.propagate_params(new_params[name])


    
    # def _recursive_update_params(self, module, new_params):
    #     self.params = new_params
    #     for name, child in module.layers.items():
    #         if isinstance(child, Module):
    #             self._recursive_update_params(child, new_params[name])
    #         else:
    #             module.params[name] = new_params[name]

