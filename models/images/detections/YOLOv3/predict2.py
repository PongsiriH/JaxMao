import numpy as np
import matplotlib.pyplot as plt
import config
from utils import cells_to_bboxes, plot_image, nms
from dataset import ImageDataset, YOLODataset, DataLoader, YOLODatasetThreeAnchors
from torchvision import datasets
from model import load_model
from YOLOv3_2.backbone.model import *
from jaxmao import Bind
import jax
import jax.numpy as jnp
from utils import prepare_data_for_map, print_boxes, calculate_mAP, plot_images
from pprint import pprint

dataset_path = ['/home/jaxmao/dataset/GTSDB_YOLO/images/train',  
                '/home/jaxmao/dataset/GTSDB_YOLO/images/test', 
                '/home/jaxmao/dataset/GTSDB_YOLO/images/16examples', 
                '/home/jaxmao/dataset/Road Sign Dataset/images', # 3
                "/home/jaxmao/dataset/coco128/images/train2017",
                '/home/jaxmao/dataset/GTSDB_YOLO/images/train300', # 5
                '/home/jaxmao/dataset/GTSDB_YOLO/images/val200',
                '/home/jaxmao/dataset/GTSDB_YOLO/images/calib100',
                '/home/jaxmao/dataset/BelTS_YOLO/images/train_cat', # 8
                '/home/jaxmao/dataset/BelTS_YOLO/images/test_cat', # 9
                ]

dataset_path_index = 8
NO_LABELS_LIST = [1, ]


if dataset_path_index in NO_LABELS_LIST:
    dataset = ImageDataset(
        dataset_path[dataset_path_index],
        transform=config.build_test_transform(416),
    )
else:
    dataset = YOLODatasetThreeAnchors(
        dataset_path[dataset_path_index],
        image_sizes=[(416, 416)],
        anchors=config.ANCHORS,
        builder_transform=config.build_test_transform,
        # cutmix=True,
        # builder_cutmix_transform=config.build_train_mosaic_transform
    )

loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

print('loading model...')
# model, params, states = load_model("YOLOv3_3/v1-stop_signs/best_yolov3_iter0.pkl")
model, params, states = load_model("yolov3_3_results28/best_yolov3_iter1.pkl") # r17
    
for data in loader:
    if not dataset_path_index in NO_LABELS_LIST:
        x, y = data
    else:
        try:
            x = data['image']
            y = data['bboxes']            
        except:
            x = np.array(data['image'])
    x = jnp.array(x)
    
    with Bind(model, params, states) as ctx:
        predictions = jax.jit(ctx.module)(x)
    # for i in range(3):
    #     predictions[i] = predictions[i].at[..., 1:5].set(jnp.zeros_like(predictions[i][..., 1:5]))
    print([pred.shape for pred in predictions])
    print('max, mean, q95, q99, q999')
    for i in range(3):
        sigmoid_scores = jax.nn.sigmoid(predictions[i][..., 0])
        max_ = sigmoid_scores.max().item()
        mean_ = sigmoid_scores.mean().item()
        q95 = jnp.quantile(sigmoid_scores, q=0.95)
        q99 = jnp.quantile(sigmoid_scores, q=0.99)
        q999 = jnp.quantile(sigmoid_scores, q=0.999)
        print(max_, mean_, q95, q99, q999)
        
    conf_thresh = input("Enter conf_thresh. If empty, default will be used: ")
    if not conf_thresh:  # Checks if the input is empty
        conf_thresh = config.CONF_THRESHOLD
    else:
        try:
            conf_thresh = float(conf_thresh)
        except:
            conf_thresh = [float(n) for n in conf_thresh.split()]
    
    has_label = False
    
    boxes_true_list = []
    boxes_pred_list = []
    try:
        for i, anchor in enumerate(config.ANCHORS):
            boxes_true_list.append(
                cells_to_bboxes(
                y[i], is_preds=False, anchors=anchor
            ))
        has_label = True
    except:
        pass
    
    for i, anchor in enumerate(config.ANCHORS):
        boxes_pred_list.append(
            cells_to_bboxes(
            predictions[i], is_preds=True, anchors=anchor
        ))
    
    print(len(predictions[0]), len(boxes_true_list))
    for j in range(len(predictions[0])):
        print(j, [len(box) for box in boxes_true_list])
        boxes_true = []
        boxes_pred = []
        for box in boxes_true_list:
            boxes_true += nms(box[j], iou_threshold=1, threshold=0.7, box_format="midpoint")
        for conf_idx, box in enumerate(boxes_pred_list):
            if isinstance(conf_thresh, list):
                c = conf_thresh[conf_idx]
            else:
                c = conf_thresh
            boxes_pred += nms(box[j], iou_threshold=config.NMS_IOU_THRESH, threshold=c, box_format="midpoint")
        converted_preds, converted_gts = prepare_data_for_map(boxes_pred, boxes_true)
        # converted_preds = converted_gts
        print_boxes(converted_preds, converted_gts)
        mAP = calculate_mAP(converted_preds, converted_gts)
        pprint(mAP)
        plot_images(x[j], boxes_true, boxes_pred)
        plt.show()
        plt.close()
    break