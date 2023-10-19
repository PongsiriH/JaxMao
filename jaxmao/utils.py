import pickle

def _ensure_stateful(inputs):
    if isinstance(inputs, list):
        raise TypeError('_ensure_statefule does not accept list.')
    if not isinstance(inputs, tuple) or len(inputs) != 2:
        inputs = (inputs, None)
    return inputs

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
        