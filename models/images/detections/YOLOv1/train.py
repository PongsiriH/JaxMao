import jax

from dataset import YOLOv1DataLoader
from loss import YOLO_2bboxes
from model import YOLOv1FromBackbone
from jaxmao import Bind
from jaxmao import optimizers as optim
from utils import yolo2xywh, plot_labels, get_best_prediction, apply_confidence_thresh, components2yolo, LABELS_DICT_CATEGORIZED_GTSRB
import matplotlib.pyplot as plt
import pickle, tqdm

key = jax.random.key(42)

NUM_GRIDS = 13
train_loader = YOLOv1DataLoader(
                    image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train_cat',
                    label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train_cat',
                    image_size=(416, 416),
                    num_grids=NUM_GRIDS,
                    num_classes=4,
                    batch_size=16,
                    normalize=True,
                    augment='default',
                    image_format='.jpg'
                )
# train_loader = YOLOv1DataLoader(
#                     image_dir='/home/jaxmao/dataset/Road Sign Dataset/images',
#                     label_dir='/home/jaxmao/dataset/Road Sign Dataset/labels',
#                     image_size=(200, 200),
#                     num_grids=NUM_GRIDS,
#                     num_classes=4,
#                     batch_size=16,
#                     normalize=True,
#                     augment=None,
#                     # augment='default',
#                     image_format='.png'
#                 )

# model = YOLOv1FromBackbone("backbone/results0/001_best.pkl", SBC=[NUM_GRIDS, 2, 43])
# model.set_trainable(True)
# params, states = model.init(key)

with open('result plots/yolov1_e2/001_last.pkl', 'rb') as f:
    model, params, states = pickle.load(f)
    
yolo_loss = YOLO_2bboxes()
optimizer = optim.Adam(params, 0.0005)

@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss, component_loss = yolo_loss.calculate_loss(predictions, labels) 
        return total_loss, (states, component_loss)
    (loss_value, (states, component_loss)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=True)(images, labels, params, states)
    # print(gradients)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, component_loss, params, states, optimizer_states

# with Summary(model) as ctx:
#     ctx.summary((16, 200, 200, 3))

for epoch in tqdm.tqdm(range(500), desc="epoch", position=0):
    if epoch == 5:
        optimizer.states['lr'] = 0.01
    if epoch > 5:
        optimizer.states['lr'] *= 0.95
        
    total_losses = 0.0
    for batch_idx, (images, labels) in tqdm.tqdm(enumerate(train_loader), desc="batch", position=1, leave=False):
        loss_value, component_loss, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    print('{}: loss: {}, last: {} {}'.format(epoch, total_losses/len(train_loader), loss_value, component_loss))
    train_loader.on_epoch_end()
    
    if (epoch+1) % 50 == 0:
        with open(f'result plots/yolov1_e3/001_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)

        with Bind(model, params, states) as ctx:
            predictions = ctx.module(images)
            
        num_plot_sample = 1
        for idx in range(num_plot_sample):
            image = images[idx]
            label = labels[idx]
            prediction = predictions[idx]

            best_conf, best_bbox, cls = get_best_prediction(prediction)
            best_conf = apply_confidence_thresh(best_conf, conf_thresh=0.2)
            yolo_prediction = components2yolo(best_bbox, best_conf, cls)

            plt.subplot(1, 2, 1)
            plot_labels(image, yolo2xywh(label), class_labels=LABELS_DICT_CATEGORIZED_GTSRB, num_grids=NUM_GRIDS, relative_to_grids=True)
            plt.subplot(1, 2, 2)
            plot_labels(image, yolo2xywh(yolo_prediction), class_labels=LABELS_DICT_CATEGORIZED_GTSRB, num_grids=NUM_GRIDS, relative_to_grids=True)
            plt.savefig(f'result plots/yolov1_e3/fig_epoch{epoch}_idx{idx}.jpg', dpi=600)
            plt.close()
            
with open('result plots/yolov1_e3/001_last.pkl', 'wb') as f:
    pickle.dump((model, params, states), f)

# print(type(images), type(labels))
# print(images.min(), images.max())
# print(labels.min(), labels.max())

# some_pred = jax.numpy.concatenate([labels[..., :5], labels], axis=-1)
