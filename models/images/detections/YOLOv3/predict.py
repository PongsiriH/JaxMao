import numpy as np
import matplotlib.pyplot as plt
import config
from utils import cells_to_bboxes, plot_image, nms
from dataset import ImageDataset, YOLODataset, DataLoader
from torchvision import datasets
from model import load_model
from YOLOv3_2.backbone.model import *
from jaxmao import Bind
import jax
import jax.numpy as jnp
from utils import prepare_data_for_map, print_boxes, calculate_mAP, plot_images, mean_avg_precision
from torchmetrics.detection.mean_ap import MeanAveragePrecision

dataset_path = ['/home/jaxmao/dataset/GTSDB_YOLO/images/train', 
                '/home/jaxmao/dataset/GTSDB_YOLO/images/test', 
                '/home/jaxmao/dataset/GTSDB_YOLO/images/16examples', 
                '/home/jaxmao/dataset/Road Sign Dataset/images']

dataset_path_index = 3
NO_LABELS_LIST = [1, ]


if dataset_path_index in NO_LABELS_LIST:
    dataset = ImageDataset(
        dataset_path[dataset_path_index],
        transform=config.build_test_transform(416),
    )
else:
    dataset = YOLODataset(
        dataset_path[dataset_path_index],
        image_sizes=[(416, 416)],
        anchors=config.ANCHORS,
        builder_transform=config.build_train_transform,
        cutmix=True,
        builder_cutmix_transform=config.build_train_mosaic_transform
    )

loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

print('loading model...')
model, params, states = load_model('yolov3_2_results5/best_yolov3_iter0.pkl')
# data = load_model('best_yolov3_cAnchors.pkl')
# model = data['model']
# params = data['params']
# states = data['states']
# best_params = data['best_params']


    
with Bind(model, params, states) as ctx:
    predictions = ctx.evaluate_loader(loader, mean_avg_precision, max_batches=10)
    print("eval: ", predictions)

with Bind(model, params, states) as ctx:
    # print(dir(loader.dataset.))
    predictions = ctx.predict_loader(loader, max_batches=10)
    [print(output.shape) for output in predictions]
    conf_thresh = np.mean([jax.nn.sigmoid(predictions[i][..., 0]).max().item() for i in range(3)])

    boxes_true_list = []
    boxes_pred_list = []
    has_label = False
    try:
        for i, anchor in zip(loader, config.ANCHORS):
            boxes_true_list += cells_to_bboxes(
                y[i], is_preds=False, anchors=anchor
            )
        has_label = True
    except:
        print("Dataset does not have ground truth.")

    for i, anchor in enumerate(config.ANCHORS):
        boxes_pred_list += cells_to_bboxes(
            predictions[i], is_preds=True, anchors=anchor
        )
        
    for j in range(len(predictions[0])):
        boxes_true = nms(boxes_true_list[j], iou_threshold=1, threshold=0.7, box_format="midpoint") if has_label else []
        boxes_pred = nms(boxes_pred_list[j], iou_threshold=config.NMS_IOU_THRESH, threshold=conf_thresh, box_format="midpoint")        
        converted_preds, converted_gts = prepare_data_for_map(boxes_pred, boxes_true)
        # converted_preds = converted_gts
        print_boxes(converted_preds, converted_gts)
        calculate_and_print_map(converted_preds, converted_gts)
        plot_images(x[j], boxes_true, boxes_pred,)
        plt.show()
        plt.close()
        
# for data in loader:
#     if not dataset_path_index in NO_LABELS_LIST:
#         x, y = data
#     else:
#         try:
#             x = data['image']
#             y = data['bboxes']            
#         except:
#             x = np.array(data['image'])
#     x = jnp.array(x)
    
#     with Bind(model, params, states) as ctx:
#         predictions = jax.jit(ctx.module)(x)
#     # for i in range(3):
#     #     predictions[i] = predictions[i].at[..., 1:5].set(jnp.zeros_like(predictions[i][..., 1:5]))
#     print([(jax.nn.sigmoid(predictions[i][..., 0]).max().item(), jax.nn.sigmoid(predictions[i][..., 0]).mean().item()) for i in range(3)])
#     jax.debug.breakpoint()
#     conf_thresh = input("Enter conf_thresh. If empty, default will be used: ")
#     if not conf_thresh:  # Checks if the input is empty
#         conf_thresh = config.CONF_THRESHOLD
#     else:
#         conf_thresh = float(conf_thresh)
    
#     has_label = False
    
#     boxes_true_list = []
#     boxes_pred_list = []
#     try:
#         for i, anchor in enumerate(config.ANCHORS):
#             boxes_true_list += cells_to_bboxes(
#                 y[i], is_preds=False, anchors=anchor
#             )
#         has_label = True
#     except:
#         pass
    
#     for i, anchor in enumerate(config.ANCHORS):
#         boxes_pred_list += cells_to_bboxes(
#             predictions[i], is_preds=True, anchors=anchor
#         )
        
#     for j in range(len(predictions[0])):
#         boxes_true = nms(boxes_true_list[j], iou_threshold=1, threshold=0.7, box_format="midpoint")
#         boxes_pred = nms(boxes_pred_list[j], iou_threshold=config.NMS_IOU_THRESH, threshold=conf_thresh, box_format="midpoint")        
#         converted_preds, converted_gts = prepare_data_for_map(boxes_pred, boxes_true)
#         # converted_preds = converted_gts
#         print_boxes(converted_preds, converted_gts)
#         calculate_and_print_map(converted_preds, converted_gts)
#         plot_images(x[j], boxes_true, boxes_pred,)
#         plt.show()
#         plt.close()
#     break