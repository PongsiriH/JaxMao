class Metric:
    def __init__(self, name=None):
        self.name = name

    def __call__(self, y_pred, y_true):
        if len(y_pred.shape) != 1:
            y_pred = y_pred.argmax(axis=1)
        if len(y_true.shape) != 1:
            y_true = y_true.argmax(axis=1)
        
        return self.calculate(y_pred, y_true)
    
class Accuracy(Metric):
    def __init__(self, name='accuracy'):
        self.name = name
        
    def calculate(self, y_pred, y_true):
        return (y_pred == y_true).mean()