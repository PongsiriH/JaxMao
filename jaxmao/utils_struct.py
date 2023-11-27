import jax
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_flatten_with_path
)

class PostInitialization(type):
    """
        Post initlaization of class.
        Example:
            class Layer(metaclass=PostInitialization):                    
                def _post_init(self):
                    # gather layers and params
    """
    def __call__(cls, *args, **kwargs):
        instance = super(PostInitialization, cls).__call__(*args, **kwargs)
        instance._post_init()
        return instance

@jax.tree_util.register_pytree_node_class
class VariablesDict:
    def __init__(self, _dict: dict=None):
        _dict = _dict if _dict is not None else {}
        if not all(isinstance(v, Variable) for v in _dict.values()):
            raise TypeError("All values in VariablesDict must be of type Variable")
        self._dict = _dict

    def __call__(self):
        return self._dict
    
    def __getitem__(self, name):
        return self._dict[name]
    
    def items(self):
        return self._dict.items()
    
    def values(self):
        return self._dict.values()
    
    def keys(self):
        return self._dict.keys()
        
    def get_shape(self, name):
        return self[name].shape
    
    def get_init(self, name):
        return self[name].initializer

    def get_reg(self, name):
        if self[name].regularizer is None:
            raise ValueError("regularizer of {} is None".format(name))
        return self[name].regularizer    
    
    def get_reg_value(self, name=None):
        if name is None:
            sum = 0.0
            for var in self._dict.values():
                sum += var.get_reg_value()
            return sum
        if self[name].regularizer is None:
            raise ValueError("regularizer of {} is None".format(name))
        return self._dict[name].get_reg_value()
    
    def get_value(self, name=None, as_dict=False):
        if name is None:
            return {key: var._value for key, var in self._dict.items()}
        if as_dict:
            return {name: self._dict[name]._value}
        return self._dict[name]._value
    
    def set_value(self, name, value):
        if not name in list(self._dict.keys()):
            raise ValueError("Name {} not exists".format(name))
        self._dict[name]._value = value
    
    def __setitem__(self, key, variable):
        if not isinstance(variable, Variable):
            raise TypeError("Value must be of type Variable")
        self._dict[key] = variable

    def add(self, name, variable):
        self._dict[name] = variable

    def __delitem__(self, key):
        del self._dict[key]
    
    def __str__(self) -> str:
        return str(self._dict)

    def __repr__(self) -> str:
        return str(self._dict)

    def tree_flatten(self):
        return tree_flatten(self._dict)
    
    @classmethod
    def tree_unflatten(self, metadata, data):
        return VariablesDict(_dict=tree_unflatten(metadata, data))
        

@jax.tree_util.register_pytree_node_class
class Variable:
    def __init__(self, shape: list | tuple, initializer, regularizer=None, value=None):
        if not(value is None or isinstance(value, jax.Array)):
            raise TypeError("Variable value got {}".format(type(value)))
        self.shape = shape
        self.initializer = initializer
        self.regularizer = regularizer
        self._value = value

    def init(self, key):
        self._value = self.initializer(key, self.shape)

    def __call__(self):
        return self._value

    def get_init(self):
        return self.initializer

    def get_shape(self):
        return self.shape
    
    def get_value(self):
        return self._value
    
    def get_reg(self):
        if self.regularizer is None:
            raise ValueError("regularizer is None.")
        return self.regularizer
    
    def get_reg_value(self):
        if self.regularizer is None:
            return 0.0
        return self.regularizer(self._value)
    
    def get(self):
        return (self.shape, self.initializer, self.regularizer, self._value)
    
    def __str__(self) -> str:
        return str((self.shape, self.initializer, self.regularizer, self._value))

    def __repr__(self) -> str:
        return self.__str__()
        
    def tree_flatten(self):
        return ([self._value], (self.shape, self.regularizer, self.initializer))

    @classmethod
    def tree_unflatten(self, metadata, data):
        shape, initializer, regularizer = metadata
        return Variable(shape, initializer, regularizer, data[0])



class LightModule:
    """make copy of a module with only neccesary parts for forward and backward pass"""
    def __init__(self, __call__, submodules, taken_names_, params_, states_):
        self.call = __call__
        if len(submodules) == 0:
            self.submodules = dict()
        else:
            self.submodules = {name: LightModule(submodules[name].__call__, submodules[name].submodules, submodules[name].taken_names_, submodules[name].params_, submodules[name].states_) for name in submodules.keys()}
        self.taken_names_ = taken_names_
        self.params_ = params_
        self.states_ = states_

    def __call__(self, inputs):
        return self.call(inputs)
    
    def get_reg_value(self, name):
        if name not in list(self.params_.keys()):
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_reg_value(name)
    
    def param(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_value(name)

    def state(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.states_.get_value(name)

class SummaryModule:
    """make copy of a module with only neccesary parts for forward and backward pass"""
    def __init__(self, __call__, submodules, taken_names_, params_, states_):
        self.call = __call__
        if len(submodules) == 0:
            self.submodules = dict()
        else:
            self.submodules = {name: SummaryModule(submodules[name].__call__, submodules[name].submodules, submodules[name].taken_names_, submodules[name].params_, submodules[name].states_) for name in submodules.keys()}
        self.taken_names_ = taken_names_
        self.params_ = params_
        self.states_ = states_
        
        for name in self.params_.keys():
            self.params_[name]._value = jax.numpy.zeros(self.params_[name].shape)

        for name in self.states_.keys():
            self.states_[name]._value = jax.numpy.zeros(self.states_[name].shape)
            
    def __call__(self, inputs):
        z = self.call(inputs)
        shape = z.shape
        return jax.numpy.zeros_like(z)
    
    def get_reg_value(self, name):
        if name not in list(self.params_.keys()):
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_reg_value(name)
    
    def param(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.params_.get_value(name)

    def state(self, name):
        if name not in self.taken_names_:
            raise ValueError(f"Name {name} does not exist.")
        return self.states_.get_value(name)
