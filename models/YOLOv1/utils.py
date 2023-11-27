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

def plot_labels(image, labels, class_labels=None, num_grids=None, relative_to_grids=True, 
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
        plt.grid(which="minor", color=grid_color, linestyle='-', linewidth=0.5)

    if labels is not None:
        for label in labels:
            class_id, x_center, y_center, width, height = label

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
def read_txt_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def xyxy2xywh(bboxes: list):
    """
    Parameters:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as a list [x_min, y_min, x_max, y_max].
        
    Return:
        out (list): A list of converted bounding boxes, where each bounding box is now in the format [x_center, y_center, width, height].
    """
    out = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        out.append([x_center, y_center, width, height])
    return out

def xywh2xyxy(bboxes: list, clips=False):
    """
    Parameters:
        bboxes (list): A list of bounding boxes, where each bounding box is represented as a list [x_center, y_center, width, height].
        clips (bool, optional): A flag to indicate whether to clip the bounding box coordinates. If set to True, the coordinates are clipped to the range [0, 1]. Default is False.
    
    Return:
        out (list): A list of converted bounding boxes in the format [x_min, y_min, x_max, y_max].
    """
    out = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        if clips:
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

        out.append([x1, y1, x2, y2])
    return out

def xywh2yolo(labels, num_grids, num_classes, relative_to_grids : bool=True):
    if isinstance(labels, tuple) and len(tuple) == 4:
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

def yolo2xywh(yolo_labels, threshold=0.0):
    """
    Convert YOLO label to a list of bounding boxes in normalized xywh format.

    :param yolo_labels: A numpy array of shape (S, S, C+5).
    :param threshold: Threshold for objectness score to consider a detection.
    :return: List of tuples (x_center, y_center, width, height).
    """
    S = yolo_labels.shape[0]  # Grid size
    xywh_list = []

    for grid_y in range(S):
        for grid_x in range(S):
            cell_info = yolo_labels[grid_y, grid_x]
            objectness_score = cell_info[0]
            
            if objectness_score > threshold:
                # Extract bounding box information
                x_cell, y_cell, width, height = cell_info[1:5]
                class_id = cell_info[5:].argmax()
                

                # Convert from grid-relative to image-relative coordinates
                x_center = (grid_x + x_cell) / S
                y_center = (grid_y + y_cell) / S

                xywh_list.append((class_id, x_center, y_center, width, height))

    return xywh_list

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
    
    def plot(self, column_name: str):
        if column_name not in self.column_names:
            raise ValueError("column_name not in self.column_names. where self.column_names are {}".format(self.column_names))
        x = np.arange(len(self.data[column_name]), step=0.01)
        y = self.data[column_name]
        plt.plot(x, y)
        plt.grid()