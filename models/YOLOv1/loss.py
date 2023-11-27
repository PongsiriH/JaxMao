from jaxmao.losses import Loss
import jax.numpy as jnp
import jax.lax as lax


class YOLO_2bboxes(Loss):
    def __init__(self):
        self.lambda_bbox = 5
        self.lambda_obj = 1
        self.lambda_noobj = 0.5
        self.lambda_cls = 1
        self.eps = 1e-6
    
    def calculate_loss(self, y_pred, y_true):
        iou1 = lax.stop_gradient(compute_iou(y_true[..., 1:5], y_pred[..., 1:5]))
        iou2 = lax.stop_gradient(compute_iou(y_true[..., 1:5], y_pred[..., 6:10]))
        responsible_mask = lax.stop_gradient(jnp.argmax(jnp.stack([iou1, iou2], axis=0), axis=0)[..., None, None])
        # print('responsible_mask', responsible_mask.shape)
        
        obj_exists = y_true[..., :1]
        true_bbox_sqrt = jnp.concatenate([y_true[..., 1:3], jnp.sqrt(y_true[..., 3:5]+self.eps)], axis=-1)
        true_cls = y_true[..., 5:]

        stacked_pred_conf = jnp.stack([y_pred[..., :1], y_pred[..., 5:6]], axis=-1)
        stacked_pred_bbox = jnp.stack([y_pred[..., 1:5], y_pred[..., 6:10]], axis=-1)
        
        # print('stacked_pred_conf', stacked_pred_conf.shape)
        # print('stacked_pred_bbox', stacked_pred_bbox.shape)
        
        selecteed_pred_conf = jnp.take_along_axis(stacked_pred_conf, responsible_mask, axis=-1).squeeze(-1)
        selecteed_pred_bbox = jnp.take_along_axis(stacked_pred_bbox, responsible_mask, axis=-1).squeeze(-1)

        # print('selecteed_pred_conf', selecteed_pred_conf.shape)
        # print('selecteed_pred_bbox', selecteed_pred_bbox.shape)
        
        selecteed_pred_bbox_sqrt = jnp.concatenate([selecteed_pred_bbox[..., :2], jnp.sqrt(selecteed_pred_bbox[..., 2:]+self.eps)], axis=-1)
        pred_cls = y_pred[..., 10:]
        # print(selecteed_pred_conf.shape, selecteed_pred_bbox.shape, pred_cls.shape)
        print('obj_exists', obj_exists)
        print('true_bbox_sqrt', true_bbox_sqrt)
        bbox_loss = obj_exists * (
            jnp.square(true_bbox_sqrt - selecteed_pred_bbox_sqrt)
        )
        obj_loss = obj_exists * (
            jnp.square(selecteed_pred_conf - 1)
        )
        noobj_loss = (1-obj_exists) * (
            jnp.square(y_pred[..., :1] - 0)
            + jnp.square(y_pred[..., 5:6] - 0)
        )
        cls_loss = obj_exists * (
            jnp.square(pred_cls - true_cls)
        )
        total_loss = jnp.sum(
            self.lambda_bbox * bbox_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * noobj_loss
            + self.lambda_cls * cls_loss
        )
        
        components_loss = {
            'bbox_loss' : jnp.sum(bbox_loss),
            'obj_loss' : jnp.sum(obj_loss),
            'noobj_loss' : jnp.sum(noobj_loss),
            'cls_loss' : jnp.sum(cls_loss),
        }
        return total_loss, components_loss

def compute_iou(bbox1, bbox2, mode='xywh', eps=1e-6):
    if mode == 'xywh':
        # Converting boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        bbox1_xyxy = jnp.concatenate([bbox1[..., :2] - bbox1[..., 2:] / 2.0,
                                    bbox1[..., :2] + bbox1[..., 2:] / 2.0], axis=-1)
        bbox2_xyxy = jnp.concatenate([bbox2[..., :2] - bbox2[..., 2:] / 2.0,
                                    bbox2[..., :2] + bbox2[..., 2:] / 2.0], axis=-1)
    elif mode == 'xyxy':
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2 
    else:
        raise ValueError("Invalid mode. Expected 'xywh' or 'xyxy'.")    
    
    # Calculating the intersection areas
    intersect_mins = jnp.maximum(bbox1_xyxy[..., :2], bbox2_xyxy[..., :2])
    intersect_maxes = jnp.minimum(bbox1_xyxy[..., 2:], bbox2_xyxy[..., 2:])
    intersect_wh = jnp.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculating the union areas
    true_area = bbox1[..., 2] * bbox1[..., 3]
    pred_area = bbox2[..., 2] * bbox2[..., 3]
    union_area = true_area + pred_area - intersect_area

    # Computing the IoU
    iou = intersect_area / (union_area + eps)

    return iou

if __name__ == '__main__':
    pass