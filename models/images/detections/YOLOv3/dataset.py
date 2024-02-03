"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from re import T
import config
import numpy as np
import os
import pandas as pd
import torch
import jax.numpy as jnp
import random

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from utils import (
    read_txt_annotations,
    xyxy2xywh,
    xywh2xyxy,
    xywh2yolo,
    yolo2xywh,
    iou_width_height,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        img_dir,
        anchors,
        labels_dir=None,
        image_sizes=[(416, 416)], # currently only support square sizes.
        C=4,
        change_size_interval=None,
        cutmix=False,
        mosaic=False,
        builder_transform=None,
        builder_mosaic_transform=None,
        builder_cutmix_transform=None,
    ):
        # if any image_sizes not divisible, by model strides, round 
        self.img_dir = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.ppm')]
        if labels_dir is None:
            self.label_dir = [p.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.ppm', '.txt') for p in self.img_dir]
        else:
            raise NotImplementedError("Have not implemented.")
            self.label_dir = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.txt')]
        self.image_sizes = image_sizes
        self.num_grids = [(image_size[0] // 32, image_size[0] // 16, image_size[0] // 8) for image_size in image_sizes]
        self.anchors = np.array([anchors[0], anchors[1], anchors[2]])  # for all 3 scales
        # self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = 3
        # self.num_anchors_per_scale = self.num_anchors
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.cutmix = 0 if cutmix == False else cutmix
        self.mosaic = 0 if mosaic == False else mosaic
        self.prob_cutmix = self.cutmix
        self.prob_mosaic = self.mosaic
        self.builder_transform = builder_transform
        self.builder_mosaic_transform = builder_mosaic_transform
        self.builder_cutmix_transform = builder_cutmix_transform
        if builder_mosaic_transform is not None:
            self.mosaic = True
        if builder_cutmix_transform is not None:
            self.cutmix = True
        self.transform = None
        self.mosaic_transform = None
        self.cutmix_transform = None
        
        self.change_size_interval = change_size_interval
        self.current_image_size_index = 0  
        self.image_size = self.image_sizes[self.current_image_size_index]
        self.S = self.num_grids[self.current_image_size_index]
        self.update_transforms()
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        if self.cutmix and np.random.binomial(1, p=self.prob_cutmix):
            indices = [random.randint(0, len(self.img_dir) - 1) for _ in range(2)]
            img_paths = [self.img_dir[i] for i in indices]
            label_paths = [self.label_dir[i] for i in indices]
            image, labels = cutmix_object_detection_using_paths(img_paths, label_paths, alpha=1.0, format="xywh", transform=self.builder_cutmix_transform, return_dict=True)
        elif self.mosaic and np.random.binomial(1, p=self.prob_mosaic):
            indices = [random.randint(0, len(self.img_dir) - 1) for _ in range(4)]
            img_paths = [self.img_dir[i] for i in indices]
            label_paths = [self.label_dir[i] for i in indices]
            image, labels = create_mosaic(img_paths, label_paths, self.image_size, self.mosaic_transform, return_dict=True)
        
        else:
            label_path = self.label_dir[index]
            labels = read_txt_annotations(label_path, return_dict=True)
            img_path = self.img_dir[index]
            image = np.array(Image.open(img_path).convert("RGB").resize(self.image_size))
        
        labels['xywh'] = [xyxy2xywh(xywh2xyxy(box, clips=True)) for box in labels['xywh']]
        # lbefore = labels['class_labels']
        # print(labels['xywh'])
        if self.transform:
            augmentations = self.transform(image=image, bboxes=labels['xywh'], class_labels=labels['class_labels'])
            image = augmentations["image"]
            labels['xywh'] = augmentations["bboxes"]
            labels['class_labels'] = augmentations["class_labels"]
        labels['xywh'] = [xyxy2xywh(xywh2xyxy(bbox, clips=True)) for bbox in labels['xywh']]
        # lafter = labels['class_labels']
        # print(f'before-after: {lbefore}:::::{lafter}')
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [np.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]
        for class_label, box in zip(labels['class_labels'], labels['xywh']):
            iou_anchors = np.concatenate(
                [iou_width_height(np.array(box[2:]), scale_anchors) for scale_anchors in self.anchors]
            )
            sorted_anchors_indices = iou_anchors.argsort(axis=0)[::-1]
            best_anchor_idx = np.argmax(iou_anchors)
            scale_idx = int(best_anchor_idx) // 3
            best_anchor_scale_idx = int(best_anchor_idx) % 3
            x, y, width, height = box
            S = self.S[scale_idx]
            i, j = int(S * y), int(S * x)  # which cell
            # print('best_scale_anchor', best_scale_anchor)
            # print('target[scale_idx]', targets[scale_idx].shape)
            targets[scale_idx][best_anchor_scale_idx, i, j, 0] = 1
            x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
            width_cell, height_cell = (
                width * S,
                height * S,
            )  # can be greater than 1 since it's relative to cell
            box_coordinates = np.array(
                [x_cell, y_cell, width_cell, height_cell]
            )
            targets[scale_idx][best_anchor_scale_idx, i, j, 1:5] = box_coordinates
            targets[scale_idx][best_anchor_scale_idx, i, j, 5] = int(class_label)

            for anchor_idx in sorted_anchors_indices[1:]:
                scale_idx = int(anchor_idx) // 3
                anchor_scale_idx = int(anchor_idx) % 3
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                if iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_scale_idx, i, j, 0] = -1  # ignore prediction
        # [print('hello: ', target[target[..., 0] > 0.5]) for target in targets]
        return np.array(image), tuple([np.array(target) for target in targets])

    def update_transforms(self):
        if self.builder_transform is not None:
            self.transform = self.builder_transform(image_size=self.image_size[0])
        if self.builder_mosaic_transform is not None:
            self.mosaic_transform = self.builder_mosaic_transform(image_size=self.image_size[0]) 
        if self.builder_cutmix_transform is not None:
            self.cutmix_transform = self.builder_cutmix_transform(image_size=self.image_size[0]) 

    def update_image_size(self, epoch):
        if self.change_size_interval is not None:
            index = (epoch // self.change_size_interval) % len(self.image_sizes)
            if index != self.current_image_size_index:
                self.current_image_size_index = index
                self.image_size = self.image_sizes[self.current_image_size_index]
                self.S = self.num_grids[self.current_image_size_index]
                self.update_transforms()
                
class YOLODatasetThreeAnchors(Dataset):
    def __init__(
        self,
        img_dir,
        anchors,
        labels_dir=None,
        image_sizes=[(416, 416)], # currently only support square sizes.
        C=4,
        change_size_interval=None,
        cutmix=False,
        mosaic=False,
        builder_transform=None,
        builder_mosaic_transform=None,
        builder_cutmix_transform=None,
    ):
        # if any image_sizes not divisible, by model strides, round 
        self.img_dir = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.ppm')]
        if labels_dir is None:
            self.label_dir = [p.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.ppm', '.txt') for p in self.img_dir]
        else:
            raise NotImplementedError("Have not implemented.")
            self.label_dir = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.txt')]
        self.image_sizes = image_sizes
        self.num_grids = [(image_size[0] // 32, image_size[0] // 16, image_size[0] // 8) for image_size in image_sizes]
        self.anchors = jnp.array(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.cutmix = cutmix
        self.mosaic = mosaic
        self.builder_transform = builder_transform
        self.builder_mosaic_transform = builder_mosaic_transform
        self.builder_cutmix_transform = builder_cutmix_transform
        if builder_mosaic_transform is not None:
            self.mosaic = True
        if builder_cutmix_transform is not None:
            self.cutmix = True
        self.transform = None
        self.mosaic_transform = None
        self.cutmix_transform = None
        
        self.change_size_interval = change_size_interval
        self.current_image_size_index = 0  
        self.image_size = self.image_sizes[self.current_image_size_index]
        self.S = self.num_grids[self.current_image_size_index]
        self.update_transforms()
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        if self.cutmix and np.random.binomial(1, p=0.4):
            indices = [random.randint(0, len(self.img_dir) - 1) for _ in range(2)]
            img_paths = [self.img_dir[i] for i in indices]
            label_paths = [self.label_dir[i] for i in indices]
            image, labels = cutmix_object_detection_using_paths(img_paths, label_paths, alpha=1.0, format="xywh", transform=self.builder_cutmix_transform, return_dict=True)
        elif self.mosaic and np.random.binomial(1, p=0.1):
            indices = [random.randint(0, len(self.img_dir) - 1) for _ in range(4)]
            img_paths = [self.img_dir[i] for i in indices]
            label_paths = [self.label_dir[i] for i in indices]
            image, labels = create_mosaic(img_paths, label_paths, self.image_size, self.mosaic_transform, return_dict=True)
        
        else:
            label_path = self.label_dir[index]
            labels = read_txt_annotations(label_path, return_dict=True)
            img_path = self.img_dir[index]
            image = np.array(Image.open(img_path).convert("RGB").resize(self.image_size))
        
        labels['xywh'] = [xyxy2xywh(xywh2xyxy(box, clips=True)) for box in labels['xywh']]
        # lbefore = labels['class_labels']
        # print(labels['xywh'])
        if self.transform:
            augmentations = self.transform(image=image, bboxes=labels['xywh'], class_labels=labels['class_labels'])
            image = augmentations["image"]
            labels['xywh'] = augmentations["bboxes"]
            labels['class_labels'] = augmentations["class_labels"]
        labels['xywh'] = [xyxy2xywh(xywh2xyxy(bbox, clips=True)) for bbox in labels['xywh']]
        # lafter = labels['class_labels']
        # print(f'before-after: {lbefore}:::::{lafter}')
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [np.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for class_label, box in zip(labels['class_labels'], labels['xywh']):
            iou_anchors = iou_width_height(np.array(box[2:]), self.anchors)
            anchor_indices = iou_anchors.argsort(axis=0)[::-1]
            x, y, width, height = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = np.array(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return np.array(image), tuple([np.array(target) for target in targets])

    def update_transforms(self):
        if self.builder_transform is not None:
            self.transform = self.builder_transform(image_size=self.image_size[0])
        if self.builder_mosaic_transform is not None:
            self.mosaic_transform = self.builder_mosaic_transform(image_size=self.image_size[0]) 
        if self.builder_cutmix_transform is not None:
            self.cutmix_transform = self.builder_cutmix_transform(image_size=self.image_size[0]) 

    def update_image_size(self, epoch):
        if self.change_size_interval is not None:
            index = (epoch // self.change_size_interval) % len(self.image_sizes)
            if index != self.current_image_size_index:
                self.current_image_size_index = index
                self.image_size = self.image_sizes[self.current_image_size_index]
                self.S = self.num_grids[self.current_image_size_index]
                self.update_transforms()
            
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)
        return image
    

def create_mosaic(img_paths, label_paths, image_size, transform: A.Compose=None, return_dict=False):
    def adjust_bounding_boxes(label, scale_x, scale_y, translate_x, translate_y):
        adjusted_label = []
        for bbox in label:
            class_id, x_center, y_center, width, height = bbox
            x_center = x_center * scale_x + translate_x
            y_center = y_center * scale_y + translate_y
            width *= scale_x
            height *= scale_y
            adjusted_label.append((class_id, x_center, y_center, width, height))
        return adjusted_label

    mosaic_image = np.zeros(image_size + (3, ))
    labels = {'class_labels': [], 'xywh': []} if return_dict else []

    indices = list(range(4))
    random.shuffle(indices)
    for i, r in enumerate(indices):
        img_path = img_paths[r]
        label_path = label_paths[r]
        adj_image_size = (image_size[0] // 2, image_size[1] // 2)
        image = Image.open(img_path).convert("RGB")
        label = read_txt_annotations(label_path, return_dict=True)
        if not transform:
            image = image.resize(adj_image_size)
        image = np.array(image)

        label['xywh'] = [xyxy2xywh(xywh2xyxy(box, clips=True)) for box in label['xywh']]
        if transform:
            augmentations = transform(image=image, bboxes=label['xywh'], class_labels=label['class_labels'])
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            class_labels = augmentations["class_labels"]
        combined_labels = np.column_stack((class_labels, bboxes))
        
        x = i % 2 * adj_image_size[1]
        y = i // 2 * adj_image_size[0]
        
        mosaic_image[y:y+adj_image_size[0], x:x+adj_image_size[1]] = image

        scale_x, scale_y = adj_image_size[1] / image_size[1], adj_image_size[0] / image_size[0]
        translate_x, translate_y = x / image_size[1], y / image_size[0]
        adjusted_label = adjust_bounding_boxes(combined_labels, scale_x, scale_y, translate_x, translate_y)
        
        if return_dict:
            labels['class_labels'].extend([label[0] for label in adjusted_label]),
            labels['xywh'].extend([label[1:] for label in adjusted_label])
        else:
            labels.extend(adjusted_label)

    return mosaic_image, labels    
    

def cutmix_object_detection(image1, bbox1, image2, bbox2, alpha=1.0, format="xywh", min_visibility=0.4, return_dict=False):
    """
    very inconsistent:
        when return_dict output format is xywh
        else output is xyxy
        
        also on x1,y1,x2,y2 relative to image/absolute pixel
        
    Apply CutMix to a pair of images and their bounding box annotations.

    Parameters:
    - image1, image2: NumPy arrays representing the images.
    - bbox1, bbox2: Lists of bounding boxes for the respective images.
      Each bounding box should be in the format [x_min, y_min, x_max, y_max, class_id].
    - alpha: Hyperparameter for the beta distribution used to sample the cut size.

    Returns:
    - mixed_image: The resulting CutMix image.
    - mixed_bbox: The adjusted bounding boxes for the mixed image.
    """
    bbox1 = bbox1.copy()
    bbox2 = bbox2.copy()
    if format == "xywh":
        for i, bbox in enumerate(bbox1):
            bbox1[i] = [bbox[0]] + list(xywh2xyxy(bbox[1:]))
            
        for i, bbox in enumerate(bbox2):
            bbox2[i] = [bbox[0]] + list(xywh2xyxy(bbox[1:]))

            
    # Sample the Beta distribution to get the mixing ratio
    lam = np.random.beta(alpha, alpha)
    
    # Calculate the mix region (rectangle coordinates)

    height, width = image1.shape[:2]
    rx, ry = np.random.randint(0, width), np.random.randint(0, height)
    rw, rh = int(width * np.sqrt(1 - lam)), int(height * np.sqrt(1 - lam))
    x1, y1, x2, y2 = max(rx - rw // 2, 0), max(ry - rh // 2, 0), min(rx + rw // 2, width), min(ry + rh // 2, height)

    # Create the mixed image
    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    if format == "xywh":
        x1 /= image1.shape[1]
        y1 /= image1.shape[0]
        x2 /= image1.shape[1]
        y2 /= image1.shape[0]
        
    # Adjust bounding boxes
    
    if return_dict:
        mixed_bbox = {'class_labels': [], 'xywh': []}
    else:
        mixed_bbox = []

    for box in bbox1:
        class_id, x_min, y_min, x_max, y_max = box
        # Check if the box is completely outside the mixed region
        if x_max < x1 or x_min > x2 or y_max < y1 or y_min > y2:
            if return_dict:
                mixed_bbox: dict
                mixed_bbox['class_labels'].append(class_id)
                mixed_bbox['xywh'].append(xyxy2xywh([x_min, y_min, x_max, y_max]))
            else:
                mixed_bbox.append(box)

    for box in bbox2:
        # Adjust and include boxes from image2 that are inside the mix region
        class_id, x_min, y_min, x_max, y_max = box
        box_area = (x_max - x_min) * (y_max - y_min)

        x_min_clipped = np.clip(x_min, x1, x2)
        y_min_clipped = np.clip(y_min, y1, y2)
        x_max_clipped = np.clip(x_max, x1, x2)
        y_max_clipped = np.clip(y_max, y1, y2)
        
        clipped_area = (x_max_clipped - x_min_clipped) * (y_max_clipped - y_min_clipped)
        visibility = clipped_area / box_area
        
        # c = not (x_min > x2 or x_max < x1 or y_min > y2 or y_max < y1)
        # # print('x', x1, x2, y1, y2)
        # # print("conditions: ", c, x_min > x2, x_max < x1, y_min > y2, y_max < y1)
        # # Check if the box still has a valid size after clipping
        if visibility > min_visibility and not (x_min_clipped > x2 or x_max_clipped < x1 or y_min_clipped > y2 or y_max_clipped < y1) :
            if return_dict:
                mixed_bbox: dict
                mixed_bbox['class_labels'].append(class_id)
                mixed_bbox['xywh'].append(xyxy2xywh([x_min, y_min, x_max, y_max]))
            else:
                mixed_bbox.append([class_id, x_min, y_min, x_max, y_max])

    return mixed_image, mixed_bbox

def convert_to_list(xywh: list, class_labels: list):
    converted_list = []
    for bbox, class_label in zip(xywh, class_labels):
        x_center, y_center, width, height = bbox
        converted_bbox = [class_label, x_center, y_center, width, height]
        converted_list.append(converted_bbox)
    return converted_list

def cutmix_object_detection_using_paths(img_paths, target_paths, alpha=1.0, format="xywh", return_dict=False, transform=None, min_visibility=0.4):
    image1 = np.array(Image.open(img_paths[0]).convert("RGB"))
    image2 = np.array(Image.open(img_paths[1]).convert("RGB").resize(image1.shape[:2][::-1]))
    bbox1 = read_txt_annotations(target_paths[0], clip=True, return_dict=True)
    bbox2 = read_txt_annotations(target_paths[1], clip=True, return_dict=True)
    
    # print(image1.shape, image2.shape)
    if transform:
        t = transform(image1.shape[0])
        aug1 = t(image=image1, bboxes=bbox1['xywh'], class_labels=bbox1['class_labels'])
        aug2 = t(image=image2, bboxes=bbox2['xywh'], class_labels=bbox2['class_labels'])

        image1, xywh1, class_labels1 = aug1['image'], aug1['bboxes'], aug1['class_labels']
        image2, xywh2, class_labels2 = aug2['image'], aug2['bboxes'], aug2['class_labels']
        
    bbox1 = convert_to_list(xywh1, class_labels1)
    bbox2 = convert_to_list(xywh2, class_labels2)
    return cutmix_object_detection(image1, bbox1, image2, bbox2, alpha=alpha, format=format, return_dict=return_dict, min_visibility=min_visibility)

def test():
    import matplotlib.pyplot as plt
    from utils import cells_to_bboxes, plot_image, nms
    dataset_path = ['/home/jaxmao/dataset/GTSDB_YOLO/images/train300', '/home/jaxmao/dataset/Road Sign Dataset/images', '/home/jaxmao/dataset/BelTS_YOLO/images/train_cat']
    
    anchors = config.ANCHORS

    transform = config.build_train_transform

    # dataset = YOLODataset(
    #     dataset_path[0],
    #     anchors=anchors,
    #     builder_transform=transform,
    #     cutmix=True,
    #     builder_cutmix_transform=config.build_train_mosaic_transform
    # )

    dataset = YOLODataset(
			img_dir=dataset_path[2],
			image_sizes=[(416, 416)],
			change_size_interval=5,
			C=4,
			anchors=config.ANCHORS,
			builder_transform=config.build_train_transform,
			# cutmix=True,
			# builder_cutmix_transform=config.build_train_mosaic_transform,
			mosaic=True,
			builder_mosaic_transform=config.build_train_mosaic_transform
		)
    
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []
        # print('y: ', [y_scale[..., 0] for y_scale in y])
        for i in range(y[0].shape[1]):
            anchor = config.ANCHORS[i]
            # print('anchor: ', anchor.shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, anchors=anchor
            )[0]
        print('boxes before nms', np.shape(boxes))
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print('boxes', boxes)
        plot_image(x[0], boxes)
        plt.waitforbuttonpress()
        plt.close()

if __name__ == "__main__":
    test()