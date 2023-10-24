from jax.tree_util import register_pytree_node

class RecursiveDict(dict):
    """
        update params and state inplace keeping same id.
    """
    @staticmethod
    def recursive_update(original, new):
        for key, value in new.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                RecursiveDict.recursive_update(original[key], value)
            else:
                original[key] = value

    def __setitem__(self, key, value):
        if key in self and isinstance(self[key], dict) and isinstance(value, dict):
            self.recursive_update(self[key], value)
        else:
            super().__setitem__(key, value)

# Register RecursiveDict as a pytree
def flatten_recursive_dict(d):
    if not d:
        return (), ()
    keys, values = zip(*sorted(d.items()))
    return values, keys

def unflatten_recursive_dict(keys, values):
    return RecursiveDict(zip(keys, values))

register_pytree_node(RecursiveDict, flatten_recursive_dict, unflatten_recursive_dict)

class PostInitialization(type):
    """
        Post initlaization of class.
        Example:
            class Layer(metaclass=PostInitialization)...
                def __init__(self..):
                    ..
                def post_initialization(self):
                    # gather layers and params
    """
    def __call__(cls, *args, **kwargs):
        instance = super(PostInitialization, cls).__call__(*args, **kwargs)
        instance.post_initialization()
        return instance
    
def _check_dict_ids(dict1, dict2):
    if id(dict1) != id(dict2):
        return False

    for key in dict1:
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                if not _check_dict_ids(dict1[key], dict2[key]):
                    return False
            elif isinstance(dict1[key], dict) or isinstance(dict2[key], dict):
                return False  # One is a dict and the other is not
        else:
            return False  # Key is missing in dict2
    return True

if __name__ == '__main__':
    import jax.numpy as jnp
    # Initialize RecursiveDict instances for original and new params
    original_params = RecursiveDict({
        'layer1': {
            'sublayer1': {'weights': jnp.array([1, 2]), 'biases': jnp.array([3])}
        }
    })

    new_params = RecursiveDict({
        'layer1': {
            'sublayer1': {'weights': jnp.array([10, 11]), 'biases': jnp.array([12])}
        }
    })

    # Get the id of the original sublayer1 dictionary
    original_id = id(original_params['layer1']['sublayer1'])

    # Update the original params with new params
    original_params['layer1'] = new_params['layer1']

    # Get the id of the updated sublayer1 dictionary
    updated_id = id(original_params['layer1']['sublayer1'])

    # Check if the ids match
    print("Original id:", original_id)
    print("Updated id:", updated_id)
    print("Ids match:", original_id == updated_id)
