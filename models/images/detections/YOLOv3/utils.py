import json
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LABELS_DICT_CATEGORIZED_GTSRB = dict({0:'prohibitory', 1:'mandatory', 2:'danger', 3:'other'})

"""
Plot utils
"""
def get_best_prediction(prediction):
    S = prediction.shape[0]
    confidences = np.array([prediction[..., :1], prediction[..., 5:6]])
    bboxes = np.array([prediction[..., 1:5], prediction[..., 6:10]])
    cls = np.array(prediction[..., 10:])
    
    # conf_idx = np.squeeze(np.argmax(confidences, axis=0), axis=-1)
    conf_idx = np.where(prediction[..., :1] > prediction[..., 5:6], 0, 1)

    best_conf = np.zeros((S, S, 1))
    best_bbox = np.zeros((S, S, 4))
    for i in range(S):
        for j in range(S):
            idx = conf_idx[i, j]
            best_conf[i, j, :] = confidences[idx, i, j, :]
            best_bbox[i, j, :] = bboxes[idx, i, j, :]
            
    return best_conf, best_bbox, cls

def apply_confidence_thresh(conf, conf_thresh=0.9):
    return np.where(conf >= conf_thresh, 1, 0)

def components2yolo(bbox, conf, cls):
    selected_bbox = np.array(bbox * conf)
    selected_bbox = np.where(selected_bbox < 0, 0, selected_bbox)
    # print(selected_bbox.shape, conf.shape, cls.shape, np.concatenate([selected_bbox, cls], axis=-1).shape)
    return np.concatenate([conf, selected_bbox, cls], axis=-1)

def plot_labels(image, classes, xywh, class_labels=None, num_grids=None, relative_to_grids=True, 
                 title='', show_grid=True, grid_color='lightgray',
                ):
    """
    Plots bounding boxes on the image.

    :param image: The image as a numpy array.
    :param labels: List of tuples (class_id, x_center, y_center, width, height).
    :param class_labels: Optional dictionary mapping class IDs to class names.
    """
    if relative_to_grids and num_grids is None:
        raise ValueError('relative to grids but num_grids is not provided')
        
    plt.imshow(image)
    plt.title(title)
    img_height, img_width = image.shape[:2]

    if show_grid and num_grids is not None:
        plt.xticks([i*(img_width/num_grids) for i in range(num_grids)], minor=True)
        plt.yticks([i*(img_height/num_grids) for i in range(num_grids)], minor=True)
        plt.grid(which="minor", color=grid_color, linestyle='-', linewidth=0.2)

    if classes is not None and xywh is not None:
        for class_id, (x_center, y_center, width, height) in zip(classes, xywh):
            class_id = int(class_id)
            # Convert normalized coordinates to pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            if relative_to_grids:
                width /= num_grids
                height /= num_grids
            # Calculate the top-left corner of the rectangle
            x_min = x_center - width / 2
            y_min = y_center - height / 2

            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

            # Add the rectangle to the Axes
            plt.gca().add_patch(rect)

            # Optionally add class labels
            if class_labels and class_id in class_labels:
                plt.text(x_min, y_min, class_labels[class_id], color='blue', fontsize=12)

"""
Data loader utils
"""
def read_txt_annotations(file_path, clip=False, return_dict=False):
    annotations = {'class_labels': [], 'xywh': []} if return_dict else []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                if clip:
                    x_center, y_center, width, height = xyxy2xywh(xywh2xyxy([x_center, y_center, width, height], clips=True))
                if return_dict:
                    annotations['class_labels'].append(class_id)
                    annotations['xywh'].append([x_center, y_center, width, height])
                else:
                    annotations.append([class_id, x_center, y_center, width, height])
    return annotations

def iou_width_height(bbox1, bbox2, eps=1e-6):
    """https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/utils.py"""
    intersection = np.minimum(bbox1[..., 0], bbox2[..., 0]) * np.minimum(bbox1[..., 1], bbox2[..., 1])
    union = bbox1[..., 0] * bbox1[..., 1] + bbox2[..., 0] * bbox2[..., 1] - intersection
    return intersection / (union + eps)

def xyxy2xywh(bboxes: list):
    """
    Parameters:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as a list [x_min, y_min, x_max, y_max].
        
    Return:
        out (list): A list of converted bounding boxes, where each bounding box is now in the format [x_center, y_center, width, height].
    """
    x_min, y_min, x_max, y_max = bboxes
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    out = (x_center, y_center, width, height)
    return out

def xywh2xyxy(bboxes: list, clips=False):
    """
    Parameters:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as a list [x_center, y_center, width, height].
        clips (bool, optional): A flag to indicate whether to clip the bounding box coordinates. If set to True, the coordinates are clipped to the range [0, 1]. Default is False.
    
    Return:
        out (list): A list of converted bounding boxes in the format [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = bboxes
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    if clips:
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))

    out = (x1, y1, x2, y2)
    return out

def xywh2yolo(labels, num_grids, num_classes, relative_to_grids : bool=True):
    if isinstance(labels, tuple) and len(labels) == 4:
        labels = [labels]

    grid = np.zeros((num_grids, num_grids, 5 + num_classes))

    for label in labels:
        class_id, x_center, y_center, width, height = label
        grid_x = int(x_center * num_grids)
        grid_y = int(y_center * num_grids)

        x_cell = x_center * num_grids - grid_x
        y_cell = y_center * num_grids - grid_y
        if relative_to_grids:
            width *= num_grids
            height *= num_grids

        if 0 <= grid_x < num_grids and 0 <= grid_y < num_grids:
            if grid[grid_y, grid_x, 0] == 0:
                grid[grid_y, grid_x, 1:5] = [x_cell, y_cell, width, height]
                grid[grid_y, grid_x, 0] = 1  # Objectness score
                grid[grid_y, grid_x, 5 + class_id] = 1  # One-hot encoded class label
            
    return grid

def yolo2xywh(yolo_labels, threshold=0.0, apply_one_hot=False, num_classes=-1):
    """
    Convert YOLO label to a list of bounding boxes in normalized xywh format.

    :param yolo_labels: A numpy array of shape (num_anchors, S, S, C+5).
    :param threshold: Threshold for objectness score to consider a detection.
    :return: List of tuples (class_id, x_center, y_center, width, height).
    """
    num_anchors = yolo_labels.shape[0]  # Number of anchors
    S = yolo_labels.shape[1]            # Grid size
    cls_list = []
    conf_list = []
    xywh_list = []

    for anchor in range(num_anchors):
        for grid_y in range(S):
            for grid_x in range(S):
                cell_info = yolo_labels[anchor, grid_y, grid_x, :]
                conf_score = cell_info[0]

                if conf_score > threshold:
                    # Extract bounding box information
                    x_cell, y_cell, width, height = cell_info[1:5]
                    class_score = jax.nn.one_hot(cell_info[5], num_classes=num_classes) if apply_one_hot else cell_info[5:]

                    # Convert from grid-relative to image-relative coordinates
                    x_center = (grid_x + x_cell) / S
                    y_center = (grid_y + y_cell) / S

                    cls_list.append(class_score)
                    conf_list.append(conf_score)
                    xywh_list.append((x_center, y_center, width, height))
    return cls_list, conf_list, xywh_list

class Results:
    def __init__(self, column_names: list):
        self.column_names = column_names
        self.data = {column_name: [] for column_name in self.column_names}
    
    def append(self, data: list):
        if len(data) != len(self.column_names):
            raise ValueError("len(data) != len(self.column_names). where self.column_names are {}".format(self.column_names))
        
        if isinstance(data, list):
            for idx, column_name in enumerate(self.column_names):
                self.data[column_name].append(data[idx])
        elif isinstance(data, dict):
            for column_name in self.column_names:
                self.data[column_name].append(data[column_name]) 
    
    def to_json(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.data, json_file)
    
    def plot(self, column_name: str):
        if column_name not in self.column_names:
            raise ValueError("column_name not in self.column_names. where self.column_names are {}".format(self.column_names))
        x = np.arange(len(self.data[column_name]))
        y = self.data[column_name]
        plt.plot(x, y)
        plt.grid()
        
        
        
def cells_to_bboxes(predictions: np.ndarray, anchors: np.ndarray, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    predictions = np.array(predictions)
    batch_size = predictions.shape[0]
    num_anchors = predictions.shape[1]
    S = predictions.shape[2]
    anchors = np.reshape(anchors, (1, num_anchors, 1, 1, 2))

    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors * S
        box_predictions[..., :2] = 2 * (np.array(jax.nn.sigmoid(box_predictions[..., :2]))) - 0.5
        box_predictions[..., 2:] = np.array(np.square(2 * jax.nn.sigmoid(box_predictions[..., 2:]))) * anchors
        # box_predictions[..., 2:] = anchors * np.exp(box_predictions[..., 2:])
        best_class = jnp.argmax(predictions[..., 5:], axis=-1)[..., None]
        scores = jax.nn.sigmoid(predictions[..., 0:1]) # * jax.nn.softmax(jnp.max(predictions[..., 5:], axis=-1)[..., None])
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = jnp.tile(jnp.arange(S), [predictions.shape[0], 3, 1, 1, 1]).reshape(batch_size, 3, 1, S, 1)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.transpose(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = jnp.concatenate((best_class, scores, x, y, w_h), axis=-1).reshape(batch_size, num_anchors * S * S, 6)
    converted_bboxes = np.array(converted_bboxes).tolist()
    return list(converted_bboxes)

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    # class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    # class_labels = [str(n) for n in range(80)]
    # colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    # fig, ax = plt.subplots(1)
    # Display the image
    plt.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            # edgecolor=colors[int(class_pred)],
            edgecolor="blue",
            facecolor="none",
        )
        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            # s=class_labels[int(class_pred)],
            s=class_pred,
            color="white",
            verticalalignment="top",
            bbox={"facecolor": "black", "pad": 0},
            # bbox={"color": colors[int(class_pred)], "pad": 0},
            # bbox={"color": int(class_pred), "pad": 0},
        )
        

def compute_iou(bbox1, bbox2, box_format='corners', eps=1e-6, stop_gradient=False):
    if box_format == 'corners':
        # Converting boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        bbox1_xyxy = jnp.concatenate([bbox1[..., :2] - bbox1[..., 2:] / 2.0,
                                    bbox1[..., :2] + bbox1[..., 2:] / 2.0], axis=-1)
        bbox2_xyxy = jnp.concatenate([bbox2[..., :2] - bbox2[..., 2:] / 2.0,
                                    bbox2[..., :2] + bbox2[..., 2:] / 2.0], axis=-1)
    elif box_format == 'midpoint':
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2 
    else:
        raise ValueError("Invalid box_format. Expected 'corners' or 'midpoint'.")    
    
    # Calculating the intersection areas
    intersect_mins = jnp.maximum(bbox1_xyxy[..., None, :2], bbox2_xyxy[None, ..., :2])
    intersect_maxes = jnp.minimum(bbox1_xyxy[..., None, 2:], bbox2_xyxy[None, ..., 2:])
    intersect_wh = jnp.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculating the union areas
    area1  = bbox1[..., 2] * bbox1[..., 3]
    area2  = bbox2[..., 2] * bbox2[..., 3]
    union_area = area1 [:, None] + area2 [None, :] - intersect_area

    # Computing the IoU
    iou = intersect_area / (union_area + eps)
    if stop_gradient:
        iou = lax.stop_gradient(iou)
    return iou

def nms(bboxes, iou_threshold, threshold, box_format="corners", max_det=50, use_np=False):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    bboxes = np.array(bboxes)
    # print("bboxes.shape before: ", bboxes.shape)
    mask = bboxes[:, 1] >= threshold
    bboxes = bboxes[mask]
    sorted_indices  = np.argsort(bboxes[:, 1])[::-1]
    bboxes = bboxes[sorted_indices][:max_det]
    # print("bboxes.shape after: ", bboxes.shape)
    bboxes_after_nms = []
    
    if use_np:
        raise NotImplementedError("use_np has not been implemented")
        ious_mesh = compute_iou(bboxes[:, 2:], bboxes[:, 2:], box_format)
        
        index_not_same = ~np.eye(bboxes.shape[0], dtype=bool)
        iou_mask = ious_mesh > iou_threshold
        bboxes[iou_mask]
        final_mask = iou_mask & index_not_same & confidence_mask
        return bboxes[final_mask]
    else:
        counter = 0
        while len(bboxes) > 0:

            counter += 1
            chosen_box = bboxes[0]
            bboxes_after_nms.append(chosen_box.tolist())
            # print('counter: ', counter, len(bboxes), bboxes.shape, bboxes_after_nms)
            
            remaining_boxes = bboxes[1:]
            
            ious = compute_iou(chosen_box[np.newaxis, 2:], remaining_boxes[:, 2:], box_format)
            # print('ious', remaining_boxes.shape, ious.shape, ious)
            bboxes = remaining_boxes[ious[0] < iou_threshold]
            
        return bboxes_after_nms
    
    
"""Torch mAP"""

from torch import tensor
import torch

# def convert_detections_to_format(detections):
#     """
#     Converts detections from [class_pred, prob_score, x1, y1, x2, y2] format
#     to a list of dictionaries with 'boxes', 'scores', and 'labels' keys.

#     Args:
#     detections (list): List of lists, each sublist containing 
#                        [class_pred, prob_score, x1, y1, x2, y2].
    
#     Returns:
#     List[Dict]: Converted format suitable for MeanAveragePrecision.
#     """
#     converted = []
#     for det in detections:
#         class_pred, prob_score, x1, y1, x2, y2 = det
#         detection_dict = {
#             'boxes': [x1, y1, x2, y2],
#             'scores': prob_score,
#             'labels': class_pred
#         }
#         found = False
#         for item in converted:
#             if item['labels'][0] == class_pred:
#                 item['boxes'] = torch.cat((item['boxes'], tensor([[x1, y1, x2, y2]])), 0)
#                 item['scores'] = torch.cat((item['scores'], tensor([prob_score])), 0)
#                 found = True
#                 break
#         if not found:
#             converted.append({
#                 'boxes': tensor([[x1, y1, x2, y2]]),
#                 'scores': tensor([prob_score]),
#                 'labels': tensor([class_pred])
#             })
#     return converted

def convert_detections_to_format(detections):
    """
    Converts detections from [class_pred, prob_score, x1, y1, x2, y2] format
    to a list of dictionaries with 'boxes', 'scores', and 'labels' keys, where
    each key is associated with a PyTorch tensor.

    Args:
    detections (list): List of lists, each sublist containing 
                       [class_pred, prob_score, x1, y1, x2, y2].
    
    Returns:
    List[Dict]: Converted format suitable for MeanAveragePrecision.
    """
    boxes = []
    scores = []
    labels = []

    for detection in detections:
        class_pred, prob_score, x1, y1, x2, y2 = detection
        boxes.append([x1, y1, x2, y2])
        scores.append(prob_score)
        labels.append(int(class_pred))

    return [
        dict(
            boxes=torch.tensor(boxes),
            scores=torch.tensor(scores),
            labels=torch.tensor(labels),
        )
    ]
    
def prepare_data_for_map(predictions, ground_truths):
    converted_preds = convert_detections_to_format(predictions)
    converted_gts = convert_detections_to_format(ground_truths)
    return converted_preds, converted_gts

from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def print_boxes(converted_preds, converted_gts):
    print("converted_preds:")
    pprint(converted_preds)
    print("converted_gts:")
    pprint(converted_gts)

def calculate_mAP(converted_preds, converted_gts):
    metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
    metric.update(converted_preds, converted_gts)
    map_value = metric.compute()
    return map_value
        
def plot_images(image, boxes_true=None, boxes_pred=None):
    plt.subplot(1, 2, 1)
    plot_image(image, boxes_true) if boxes_true is not None else plt.imshow(image)
    plt.subplot(1, 2, 2)
    plot_image(image, boxes_pred) if boxes_pred is not None else plt.imshow(image)

def mean_avg_precision(y_pred, y_true):
    import config
    predictions = []
    ground_truths = []
    pred = []
    gts = []
    for i, anchors in enumerate(config.ANCHORS):
        # cells_to_bboxes output (N, num_anchors, S, S, 1+5) for each scales
        # pred is list of [(N, num_anchors1, S1, S1, 1+5), (N, num_anchors2, S2, S2, 1+5), (N, num_anchors3, S3, S3, 1+5)]
        pred.append(cells_to_bboxes(y_pred[i], anchors, is_preds=True))
        gts.append(cells_to_bboxes(y_true[i], anchors, is_preds=False))
    
    conf_thresh = np.mean([np.quantile(y[..., 0].ravel(), q=0.99) for y in y_pred])
    # print(len(pred), len(gts))
    for j in range(len(y_true[0])):
        # print("pred:", [len(p[j]) for p in pred])
        # print("gts:", [len(p[j]) for p in gts])
        
        converted_preds, converted_gts = prepare_data_for_map(
            nms(np.concatenate([p[j] for p in pred], axis=0), iou_threshold=0.4, threshold=conf_thresh, box_format="midpoint"), 
            nms(np.concatenate([p[j] for p in gts], axis=0), iou_threshold=1.0, threshold=0.5, box_format="midpoint")
                                                              )
        predictions.extend(converted_preds)
        ground_truths.extend(converted_gts)
        print_boxes(converted_preds, converted_gts)
    return calculate_mAP(predictions, ground_truths)