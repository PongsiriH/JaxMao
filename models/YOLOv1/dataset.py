from keras.utils import Sequence
import numpy as np
import albumentations as A
import os, cv2, warnings
from utils import (
    read_txt_annotations,
    xyxy2xywh,
    xywh2xyxy,
    xywh2yolo,
    yolo2xywh,
)

class YOLOv1DataLoader(Sequence):
    def __init__(self, image_dir, label_dir, image_size,
                 num_grids, num_classes, batch_size, normalize=True, augment='default',
                 image_format='.jpg'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.S = num_grids
        self.C = num_classes
        self.image_format = image_format
        self.normalize = normalize
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(self.image_format)]
        self.label_paths = [p.replace('images', 'labels').replace(self.image_format, '.txt') for p in self.image_paths]
        
        if isinstance(augment, A.Compose):
            self.augment = augment
        elif augment == 'default':
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.05),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
        else:
            warnings.warn("Unrecognized augmentation input. Proceeding with no augmentation.", UserWarning)
            self.augment = None

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_paths[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = []
        yolo_labels = np.zeros((self.batch_size, self.S, self.S, 5 + self.C)) # 3 anchors hard-coded
        
        for i, (img_path, label_path) in enumerate(zip(batch_image_paths, batch_label_paths)):
            # Load image and label
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.label_dir is not None:
                labels = read_txt_annotations(label_path)
                bboxes = [np.clip(label[1:], 0, 1) for label in labels]
                bboxes = xyxy2xywh(xywh2xyxy(bboxes, clips=True))
                class_labels = [label[0] for label in labels]
                
                if self.augment is not None:
                    augmented = self.augment(image=image, bboxes=bboxes, class_labels=class_labels)
                    image = augmented['image']
                    bboxes = augmented['bboxes']
                                
                updated_labels = [(class_labels[j], *bbox) for j, bbox in enumerate(bboxes)]

                # Convert labels to YOLO format
                yolo_label = xywh2yolo(updated_labels, self.S, self.C, relative_to_grids=True)
                yolo_labels[i] = np.array(yolo_label, dtype='float32')
            else:
                augmented = self.augment(image=image) if self.augment is not None else {'image': image}
                image = augmented['image']
            image = cv2.resize(image, self.image_size)      
            images.append(image)

        images = np.array(images, dtype='float32')
        
        if self.normalize:
            images = images / 255.0
        
        return (images, np.array(yolo_labels, dtype='float32')) if self.label_dir is not None else (images, [None]*self.batch_size)

    def on_epoch_end(self):
        combined = list(zip(self.image_paths, self.label_paths))
        np.random.shuffle(combined)
        self.image_paths, self.label_paths = zip(*combined)

if __name__ == '__main__':
    train_loader = YOLOv1DataLoader(
                        image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train',
                        label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train',
                        image_size=(224, 224),
                        num_grids=7,
                        num_classes=4,
                        batch_size=16,
                        normalize=True,
                        augment=None,
                        image_format='.jpg'
                    )
    for images, labels in train_loader:
        break
    print(type(images), type(labels))
    print(images.shape, labels.shape)
