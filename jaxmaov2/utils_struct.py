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
        
    def get_meta(self, name):
        return self[name].shape, self[name].initializer

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

# if __name__ == '__main__':
#     shape = [2, 3]
#     initializer = jax.nn.initializers.constant(5)
#     var = Variable(shape, initializer)
#     var.init(jax.random.key(seed=42))
#     var_flatten, var_def = tree_flatten(var)
#     print('var:', var)
#     print('flatten var:\n', var_flatten)
#     print('flatten var_def:\n', var_def)
#     print('unflatten var:\n', tree_unflatten(var_def, var_flatten).value)
    
#     vars = VariablesDict({
#         'weights' : Variable([2, 3], initializer=initializer),
#         'bias': Variable([2,], initializer=initializer)
#     })
    
#     vars_flatten, vars_def = tree_flatten(vars)
#     vars_unflatten = tree_unflatten(vars_def, vars_flatten)
#     print('vars: ', vars)
#     print('vars: ', vars_flatten)
#     print('vars: ', vars_def)
#     print('vars: ', type(vars_unflatten))
    
if __name__ == '__main__':
    class Hello:
        def __init__(self):
            self.vardict = VariablesDict()
            self.vardict.add('weights', Variable((5,2), jax.nn.celu))
            self.vardict.set_value('weights', 5)
            # print(self.vardict.get_value('weights'))
            
    class Bye:
        def __init__(self):
            self.vardict = VariablesDict()
            self.vardict.add('weights', Variable((10,2), jax.nn.celu))
            self.vardict.set_value('weights', 5)
    
    hello = Hello()
    bye = Bye()
    print(hello.vardict)
    