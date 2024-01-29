import jax.numpy as jnp
from jax import grad, jit
import jax

def mean_over_batch_size(negative_log_likelihood):
    sum_loss = jnp.sum(negative_log_likelihood)
    mean_loss = sum_loss / negative_log_likelihood.shape[0]
    return mean_loss

class Loss:
    def __init__(self, reduce_fn='mean_over_batch_size', keepdims=False, eps=1e-7):
        super().__init__()
        self.keepdims = keepdims
        self.eps = eps
        if reduce_fn is None:
            reduce_fn = lambda x: x
        elif reduce_fn == 'mean_over_batch_size':
            reduce_fn = mean_over_batch_size
        elif reduce_fn == 'mean':
            reduce_fn = jnp.mean
        elif reduce_fn == 'sum':
            reduce_fn = jnp.sum
        
        self.reduce_fn = reduce_fn

    def calculate_loss(self, y_pred, y_true, **kwargs):
        return self.call(y_pred, y_true, **kwargs)
    
    def call(self, y_pred, y_true):
        pass
    
    def __call__(self, y_pred, y_true, **kwargs):
        return self.calculate_loss(y_pred, y_true, **kwargs)

class MeanSquaredError(Loss):
    def call(self, y_pred, y_true):
        return self.reduce_fn(jnp.mean(jnp.square(y_true - y_pred), axis=-1, keepdims=self.keepdims))

class BinaryCrossEntropy(Loss):
    def call(self, y_pred, y_true, clip=True):
        if clip:
            y_pred = jnp.clip(y_pred, self.eps, 1.0 - self.eps)
        return self.reduce_fn(
                    -jnp.sum(
                        y_true * jnp.log(y_pred + self.eps) 
                        + (1 - y_true) * jnp.log(1 - y_pred + self.eps),
                    axis=-1
                    ) / y_true.shape[-1]
            )
    
class CategoricalCrossEntropy(Loss):
    def call(self, y_pred, y_true):   
        y_pred /= jnp.sum(y_pred, axis=-1, keepdims=True)
        y_pred = jnp.clip(y_pred, self.eps, 1 - self.eps)
        return (-jnp.sum(y_true * jnp.log(y_pred), axis=-1))  
        negative_log_likelihood = -y_true * jnp.log(y_pred)
        return self.reduce_fn(jnp.sum(negative_log_likelihood, axis=-1))

class BCEWithLogitsLoss(Loss):
    def call(self, logits, y_true):
        log_p = jax.nn.log_sigmoid(logits)
        log_not_p = jax.nn.log_sigmoid(-logits) # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
        return self.reduce_fn(
                jnp.mean(
                -y_true * log_p - (1. - y_true) * log_not_p, axis=-1, keepdims=self.keepdims
            )
        )

        y_pred = jax.nn.sigmoid(logits)
        y_pred_clipped = jnp.clip(y_pred, self.eps, 1 - self.eps)
        return -self.reduce_fn(y_true * jnp.log(y_pred_clipped) + (1 - y_true) * jnp.log(1 - y_pred_clipped))

class FocalLossLogit(Loss):
    def __init__(self, alpha=1.0, gamma=2.0, reduce_fn=jnp.mean):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce_fn = reduce_fn
        self.bce_logit = BCEWithLogitsLoss(reduce_fn=None, keepdims=True)

    def call(self, logits, y_true):
        """https://github.com/ultralytics/yolov5/blob/master/utils/loss.py"""
        loss = self.bce_logit(logits, y_true)
        sgm_p = jax.nn.sigmoid(logits)
        p_t = y_true * sgm_p + (1 - y_true) * (1 - sgm_p)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = loss * alpha_factor * modulating_factor
        return self.reduce_fn(focal_loss)
        
class CCEWithLogitsLoss(Loss):
    def call(self, logits, y_true):
        return self.reduce_fn(
            -jnp.sum(y_true * jax.nn.log_softmax(logits, axis=-1), axis=-1, keepdims=self.keepdims)
            )
