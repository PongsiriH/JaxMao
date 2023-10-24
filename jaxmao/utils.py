import pickle, cv2
import jax

def make_loss_function_gradable(
                method, 
                loss_fn, 
                ):
    """ 
        Example:
        funct = make_loss_function_gradable(method=model.forward, loss_fn=loss_fn)
        loss_and_grad = jax.value_and_grad(funct, argnums=0, has_aux=True)
    """
    def _aux_make_loss_function_gradable(params, x_true, y_true, state):
        y_pred, new_state = method(params, x_true, state)
        loss = loss_fn(y_pred, y_true)
        return loss, new_state
    return _aux_make_loss_function_gradable

"""
    pickle utils
"""
def save_pickle_file(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

"""
    dataset
"""
def load_GTSRB(color='rgb'):
    if not color in ['rgb', 'hsv']:
        return False
    X_train = load_pickle_file('/home/jaxmao/jaxmao/dataset/GTSRB_hsv/X_train.pkl')
    y_train = load_pickle_file('/home/jaxmao/jaxmao/dataset/GTSRB_hsv/y_train.pkl')
    X_test = load_pickle_file('/home/jaxmao/jaxmao/dataset/GTSRB_hsv/X_test.pkl')
    y_test = load_pickle_file('/home/jaxmao/jaxmao/dataset/GTSRB_hsv/y_test.pkl')

    if color == 'rgb':
        for i in range(len(X_train)):
            X_train[i] = cv2.cvtColor(X_train[i].astype('float32'), cv2.COLOR_HSV2RGB)
        for i in range(len(X_test)):
            X_test[i] = cv2.cvtColor(X_test[i].astype('float32'), cv2.COLOR_HSV2RGB)
    return (X_train, y_train), (X_test, y_test)
