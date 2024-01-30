from utils import xywh2xyxy
from jaxmao.nn.losses import Loss, BCEWithLogitsLoss, CCEWithLogitsLoss, MeanSquaredError, CategoricalCrossEntropy
import jax.numpy as jnp
import numpy as np
import jax
import jax.lax as lax

mse = MeanSquaredError(reduce_fn=None, keepdims=True)
bce_logit = BCEWithLogitsLoss(reduce_fn=None, keepdims=True)
cce = CategoricalCrossEntropy(reduce_fn=None, keepdims=True)
cce_logits = CCEWithLogitsLoss(reduce_fn=None, keepdims=True)

def loss_iou(bbox1, bbox2, box_format='corners', eps=1e-6, stop_gradient=False):
    if box_format == 'corners':
        # Converting boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        bbox1_xyxy = jnp.concatenate([bbox1[..., :2] - bbox1[..., 2:] / 2.0,
                                    bbox1[..., :2] + bbox1[..., 2:] / 2.0], axis=-1)
        bbox2_xyxy = jnp.concatenate([bbox2[..., :2] - bbox2[..., 2:] / 2.0,
                                    bbox2[..., :2] + bbox2[..., 2:] / 2.0], axis=-1)
    elif box_format == 'midpoint':
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2 
    elif box_format == 'width-height':
        bbox1_xyxy = bbox1.at[..., :2].set(0.0)
        bbox2_xyxy = bbox2.at[..., :2].set(0.0)
    else:
        raise ValueError("Invalid box_format. Expected 'corners' or 'midpoint'.")    
    
    # Calculating the intersection areas
    intersect_mins = jnp.maximum(bbox1_xyxy[..., :2], bbox2_xyxy[..., :2])
    intersect_maxes = jnp.minimum(bbox1_xyxy[..., 2:], bbox2_xyxy[..., 2:])
    intersect_wh = jnp.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculating the union areas
    area1  = bbox1[..., 2] * bbox1[..., 3]
    area2  = bbox2[..., 2] * bbox2[..., 3]
    union_area = area1 + area2- intersect_area

    # Computing the IoU
    iou = intersect_area / (union_area + eps)
    if stop_gradient:
        iou = lax.stop_gradient(iou)
    return iou

def bbox_ciou(bboxes1, bboxes2):
    """
    Calculate the Complete Intersection over Union (CIoU) of two sets of boxes.
    :param bboxes1: (numpy array) bounding boxes, Shape: [N, 4]
    :param bboxes2: (numpy array) ground truth boxes, Shape: [N, 4]
    :return: (numpy array) CIoU, Shape: [N]
    """
    # Extract coordinates for each box
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1[..., 0], bboxes1[..., 1], bboxes1[..., 2], bboxes1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2[..., 0], bboxes2[..., 1], bboxes2[..., 2], bboxes2[..., 3]

    # Calculate area of each box
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Calculate intersection area
    inter_x1 = jnp.maximum(b1_x1, b2_x1)
    inter_y1 = jnp.maximum(b1_y1, b2_y1)
    inter_x2 = jnp.minimum(b1_x2, b2_x2)
    inter_y2 = jnp.minimum(b1_y2, b2_y2)
    inter_area = jnp.maximum(inter_x2 - inter_x1, 0) * jnp.maximum(inter_y2 - inter_y1, 0)

    # Calculate union area
    union_area = b1_area + b2_area - inter_area

    # Calculate IoU
    iou = inter_area / jnp.maximum(union_area, 1e-6)

    # Calculate the center distance
    b1_center_x, b1_center_y = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_center_x, b2_center_y = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    center_distance = jnp.square(b1_center_x - b2_center_x) + jnp.square(b1_center_y - b2_center_y)

    # Calculate the diagonal of the smallest enclosing box
    enclose_x1 = jnp.minimum(b1_x1, b2_x1)
    enclose_y1 = jnp.minimum(b1_y1, b2_y1)
    enclose_x2 = jnp.maximum(b1_x2, b2_x2)
    enclose_y2 = jnp.maximum(b1_y2, b2_y2)
    enclose_diagonal = jnp.square(enclose_x2 - enclose_x1) + jnp.square(enclose_y2 - enclose_y1)

    # Aspect ratio penalty term
    v = (4 / (jnp.pi ** 2)) * jnp.square(jnp.arctan((b1_x2 - b1_x1) / jnp.maximum(b1_y2 - b1_y1, 1e-6)) - jnp.arctan((b2_x2 - b2_x1) / jnp.maximum(b2_y2 - b2_y1, 1e-6)))
    alpha = v / jnp.maximum((1 - iou + v), 1e-6)
    ciou = iou - (center_distance / jnp.maximum(enclose_diagonal, 1e-6)) - alpha * v
    return ciou

class YOLOanchorLoss_2bboxes(Loss):
    def __init__(self, lambda_box, lambda_obj, lambda_noobj, lambda_class):
        self.lambda_box = lambda_box 
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.eps = 1e-6

    def calculate_loss(self, predictions, targets, anchors):
        """
        predictions: (N, num_anchors, S, S, 4+1+num_class)
        """
        mask_obj = jnp.array(targets[..., :1] == 1, dtype=jnp.float32)
        mask_noobj = jnp.array(targets[..., :1] == 0, dtype=jnp.float32)
        mask_not_noobj = jnp.array(~(targets[..., :1] == 0), dtype=jnp.float32)
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) * np.reshape(np.array(predictions.shape[2:4]), (1, 1, 1, 1, 2))
        
        predictions = predictions.at[..., 1:3].set(2 * jax.nn.sigmoid(predictions[..., 1:3]) - 0.5)
        predictions = predictions.at[..., 3:5].set(4 * jnp.square(jax.nn.sigmoid(predictions[..., 3:5])) * anchors)
        
        no_object_loss = bce_logit(predictions[..., 0:1], targets[..., 0:1])
        no_object_loss = (mask_noobj * no_object_loss).sum()
        
        object_loss = bce_logit(predictions[..., 0:1], targets[..., 0:1])
        object_loss = (mask_obj * object_loss).sum()
        
        # targets = targets.at[..., 3:5].set(jnp.where(targets[..., 3:5] == 0, self.eps, targets[..., 3:5]))
        # targets = targets.at[..., 3:5].set(jnp.log(targets[..., 3:5] / anchors))
        box_loss = mse(predictions[..., 1:5], targets[..., 1:5])
        box_loss = (mask_obj* box_loss).sum()
        
        class_loss = cce_logits(predictions[..., 5:], jax.nn.one_hot(targets[..., 5], num_classes=predictions.shape[-1]-5))
        class_loss = (mask_not_noobj * class_loss).sum()
        
        box_loss = self.lambda_box * jnp.sum(box_loss) / targets.shape[0]
        object_loss = self.lambda_obj * jnp.sum(object_loss) / targets.shape[0]
        no_object_loss = self.lambda_noobj * jnp.sum(no_object_loss) / targets.shape[0]
        class_loss = self.lambda_class * jnp.sum(class_loss) / targets.shape[0]
        
        total_loss = box_loss + object_loss + no_object_loss + class_loss
        
        loss_components = {
            'bbox_loss' : box_loss,
            'obj_loss' : object_loss,
            'noobj_loss' : no_object_loss,
            'cls_loss' : class_loss,
        }
            
        return total_loss, loss_components