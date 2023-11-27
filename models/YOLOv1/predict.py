import jax
from jaxmao.modules import Bind
from dataset import YOLOv1DataLoader
from utils import yolo2xywh, plot_labels, get_best_prediction, apply_confidence_thresh, components2yolo, LABELS_DICT_CATEGORIZED_GTSRB
import matplotlib.pyplot as plt
import pickle

train_loader = YOLOv1DataLoader(
                    image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train',
                    label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train',
                    image_size=(416, 416),
                    num_grids=7,
                    num_classes=4,
                    batch_size=16,
                    normalize=True,
                    augment=None,
                    image_format='.jpg'
                )
for images, labels in train_loader:
    break

with open('model3.pkl', 'rb') as f:
    model, params, states = pickle.load(f)


with Bind(model, params, states) as ctx:
    predictions = ctx.module(images)
    
idx = 2
image = images[idx]
label = labels[idx]
prediction = predictions[idx]

best_conf, best_bbox, cls = get_best_prediction(prediction)
best_conf = apply_confidence_thresh(best_conf, conf_thresh=0.5)
yolo_prediction = components2yolo(best_bbox, best_conf, cls)

plt.subplot(1, 2, 1)
plot_labels(image, yolo2xywh(label), class_labels=LABELS_DICT_CATEGORIZED_GTSRB, num_grids=7, relative_to_grids=True)
plt.subplot(1, 2, 2)
plot_labels(image, yolo2xywh(yolo_prediction), class_labels=LABELS_DICT_CATEGORIZED_GTSRB, num_grids=7, relative_to_grids=True)
plt.show()

# print(predictions.shape, predictions)
