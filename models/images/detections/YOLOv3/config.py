import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.005
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.01
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], # large
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # medium
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # small
]

# ANCHORS = [
#     [(0.08455882, 0.1375), (0.06617647, 0.1125),(0.05588235, 0.0925)],
#     [(0.04779412, 0.08 ), (0.04117647, 0.06875), (0.03382353, 0.0575)],
#     [(0.02867647, 0.04875), (0.02279412, 0.0375), (0.01691176, 0.02875)]
# ]

gtsdb_anchors = [
    [[0.01691176, 0.02875   ], [0.02279412, 0.0375    ], [0.02867647, 0.04875   ]], 
    [[0.03382353, 0.0575    ], [0.04117647, 0.06875   ], [0.04779412, 0.08      ]],
    [[0.05588235, 0.0925    ], [0.06617647, 0.1125    ], [0.08455882, 0.1375    ]]
    ]

# ANCHORS = gtsdb_anchors
# ANCHORS = [
#       [(0.25, 0.25), (0.4, 0.4), (0.8, 0.8)],
#     [(0.1, 0.1), (0.15, 0.15), (0.2, 0.2)],
#     [(0.02, 0.02), (0.05, 0.05), (0.08, 0.06)],
# ]

scale = 1.1

def build_train_transform(image_size=IMAGE_SIZE):
    train_transforms = A.Compose(
        [
            A.ShiftScaleRotate(rotate_limit=5, p=0.5, border_mode=cv2.BORDER_REFLECT), 
            # A.GaussNoise(var_limit=(2, 5), per_channel=False, p=0.1),  
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            A.Resize(width=image_size, height=image_size)
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=['class_labels'],),
    )
    return train_transforms

def build_train_mosaic_transform(image_size=IMAGE_SIZE):
    train_moasic_transforms = A.Compose(
        [
            # A.LongestMaxSize(max_size=int(image_size * scale)),
            # A.PadIfNeeded(
            #     min_height=int(image_size * scale),
            #     min_width=int(image_size * scale),
            #     border_mode=cv2.BORDER_CONSTANT,
            # ),
            A.RandomSizedBBoxSafeCrop(image_size // 2, image_size // 2),
            # A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.4),
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT
                    ),
                    A.MotionBlur(p=0.2),
                ],
                p=1.0,
            ),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.ToGray(p=0.1),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=['class_labels'],),
    )
    return train_moasic_transforms

def build_test_transform(image_size=IMAGE_SIZE):
    test_transforms = A.Compose(
        [
            # A.LongestMaxSize(max_size=image_size),
            # A.PadIfNeeded(
            #     min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            # ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            A.Resize(image_size, image_size, always_apply=True)
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=['class_labels']),
    )
    return test_transforms
