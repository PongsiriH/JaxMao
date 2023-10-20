import numpy as np

class Metric:
    def __init__(self, name=None):
        self.name = name

    def __call__(self, y_pred, y_true):
        if len(y_pred.shape) == 2:
            y_pred = y_pred.argmax(axis=1)
        if len(y_true.shape) == 2:
            y_true = y_true.argmax(axis=1)
        elif len(y_true.shape) == 1:
            # Assuming y_true is label-encoded as a 1D array
            y_true = y_true.reshape(-1, 1)
        
        return self.calculate(y_pred, y_true)

"""
    Classification metrics
"""
def true_positives(y_pred, y_true):
    return np.sum((y_pred == 1) & (y_true == 1))

def false_positives(y_pred, y_true):
    return np.sum((y_pred == 1) & (y_true == 0))

def true_negatives(y_pred, y_true):
    return np.sum((y_pred == 0) & (y_true == 0))

def false_negatives(y_pred, y_true):
    return np.sum((y_pred == 0) & (y_true == 1))

class Accuracy(Metric):
    def __init__(self, name='accuracy'):
        self.name = name
        
    def calculate(self, y_pred, y_true):
        return (y_pred == y_true).mean()

class Precision(Metric):
    def __init__(self, name='precision'):
        self.name = name

    def calculate(self, y_pred, y_true):
        tp = true_positives(y_pred, y_true)
        fp = false_positives(y_pred, y_true)
        return tp / (tp + fp) if (tp + fp) > 0 else 0

class Recall(Metric):
    def __init__(self, name='recall'):
        self.name = name

    def calculate(self, y_pred, y_true):
        tp = true_positives(y_pred, y_true)
        fn = false_negatives(y_pred, y_true)
        return tp / (tp + fn) if (tp + fn) > 0 else 0

class F1Score(Metric):
    def __init__(self, name='f1_score'):
        self.name = name

    def calculate(self, y_pred, y_true):
        precision = Precision()(y_pred, y_true)
        recall = Recall()(y_pred, y_true)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

"""
    Regression metrics
"""

