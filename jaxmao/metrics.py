class Metric:
    pass

class Accuracy(Metric):
    def calculate(self, y_pred, y_true):
        return (y_pred == y_true).mean()