import jax
import jax.numpy as jnp

def _get_parameters(module):
    parameters = dict()
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True)
    for name in module.submodules:
        parameters[name] = _get_parameters(module.submodules[name])
    return parameters

def _get_states(module):
    states = dict()
    if len(module.submodules) == 0:
        return module.states_.get_value(as_dict=True)
    for name in module.submodules:
        states[name] = _get_states(module.submodules[name])
    return states

def _get_parameters_and_states(module):
    """more efficient  _get_parameters + _get_states"""
    parameters = dict()
    states = dict()
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True), module.states_.get_value(as_dict=True)
    for name in module.submodules:
        parameters[name], states[name] = _get_parameters_and_states(module.submodules[name])
    return parameters, states

def _get_states_and_regularizes(module):
    """more efficient _get_states + regularizes"""
    states = dict()
    regularizes = 0.0
    if len(module.submodules) == 0:
        return module.states_.get_value(as_dict=True), module.params_.get_reg_value()
    for name in module.submodules:
        states[name], reg = _get_states_and_regularizes(module.submodules[name])
        regularizes += reg
    return states, regularizes

def _get(module):
    """more efficient  _get_parameters + _get_states + regularizes"""
    parameters = dict()
    states = dict()
    regularizes = 0.0
    if len(module.submodules) == 0:
        return module.params_.get_value(as_dict=True), module.states_.get_value(as_dict=True), module.params_.get_reg_value()
    for name in module.submodules:
        parameters[name], states[name], regularizes = _get(module.submodules[name])
    return parameters, states, regularizes

def _update_parameters(module, params):
    if len(module.submodules) == 0:
        for key, param in params.items():
            module.params_.set_value(key, param)
        
    for name, submodule in module.submodules.items():
        _update_parameters(submodule, params[name])
        
def _update(module, params, states):
    if len(module.submodules) == 0:
        for name, param in params.items():
            module.params_.set_value(name, param)
            
        for name, state in states.items():
            module.states_.set_value(name, state)
        
    for name, submodule in module.submodules.items():
        _update(submodule, params[name], states[name])   
    
def _init(module, key):
    for name in module.params_():
        key, subkey = jax.random.split(key)
        shape, initializer = module.params_.get_shape(name), module.params_.get_init(name)
        module.params_.set_value(name, initializer(subkey, shape, 'float32'))
        
    for name in module.states_():
        key, subkey = jax.random.split(key)
        shape, initializer = module.states_.get_shape(name), module.states_.get_init(name)
        module.states_.set_value(name, initializer(subkey, shape, 'float32'))
        
    for name, submodule in module.submodules.items():
        key, subkey = jax.random.split(key)
        _init(submodule, subkey)

def _init_zero(module):
    for name in module.params_():
        shape, initializer = module.params_.get_shape(name), module.params_.get_init(name)
        module.params_.set_value(name, jnp.zeros(shape, 'float32'))
        
    for name in module.states_():
        shape, initializer = module.states_.get_shape(name), module.states_.get_init(name)
        module.states_.set_value(name, jnp.zeros(shape, 'float32'))
        
    for name, submodule in module.submodules.items():
        _init_zero(submodule)

id_counter = 0
@jax.tree_util.register_pytree_node_class
class UniqueID:
    _counters = {}
    def __init__(self, name, id=None):
        self.name = name
        if name not in UniqueID._counters:
            UniqueID._counters[name] = 0            
                
        if id is None:
            UniqueID._counters[name] += 1
            self.id = UniqueID._counters[name]
        elif isinstance(id, int):
            self.id = id
        else:
            raise ValueError('id suppose to be int. got {}'.format(id))

    def __str__(self) -> str:
        return 'UID.{}={}'.format(self.name, self.id)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return self.name < other.name
        
        return self.id < other.id

    def __le__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return self.name <= other.name
        return self.id <= other.id

    def __gt__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return self.name > other.name
        return self.id > other.id

    def __ge__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return self.name >= other.name
        return self.id >= other.id

    def __eq__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return False
        return self.id == other.id

    def __ne__(self, other):
        if not isinstance(other, UniqueID):
            return NotImplemented
        elif self.name != other.name:
            return True
        return self.id != other.id

    def __hash__(self):
        return hash((self.name, self.id))
    
    def tree_flatten(self):
        return ([self.id], self.name)
    
    @classmethod
    def tree_unflatten(self, name, id):
        return UniqueID(name, id[0])

if __name__ == '__main__':
    m1 = UniqueID('hello')
    m2 = UniqueID('hello')
    m3 = UniqueID('bye')
    print(m1)
    print(m2)    
    print(m3)
    m1_t, m1_d = jax.tree_util.tree_flatten(m3)
    print(m1_t)
    m1_re = jax.tree_util.tree_unflatten(m1_d, m1_t)
    print(m1_re)
    