import jax
import jax.numpy as jnp
import numpy as np

from jaxmao import Bind
from model import YOLOv1FromBackbone
from loss import YOLOanchorLoss_2bboxes
from dataset import YOLODataset
import jaxmao.nn.optimizers as optim
from utils import Results, yolo2xywh, plot_labels, LABELS_DICT_CATEGORIZED_GTSRB
import pickle
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

key = jax.random.key(42)

NUM_GRIDS = [26, 26, 26]
NUM_CLASSES = 4
# train_loader = YOLODataset(
#                     image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train',
#                     label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train',
#                     image_size=(400, 400),
#                     num_grids=13,
#                     num_classes=4,
#                     batch_size=16,
#                     normalize=True,
#                     augment='default',
#                     image_format='.jpg'
#                 )
train_dataset = YOLODataset(
                    img_dir='/home/jaxmao/dataset/Road Sign Dataset/images',
                    image_size=(416, 416),
                    S=NUM_GRIDS,
                    C=NUM_CLASSES,
                    anchors=config.ANCHORS
                )
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

model = YOLOv1FromBackbone(num_classes=NUM_CLASSES)
model.set_trainable(True)
params, states = model.init(key)
    
yolo_loss = YOLOanchorLoss_2bboxes()
optimizer = optim.Adam(params, 1e-4)
epoch_result = Results(['total_losses', 'loss_value', 'bbox_loss', 'obj_loss', 'noobj_loss', 'cls_loss', 'lr'])

@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        
        total_loss = 0.0
        component_loss = {'bbox_loss': 0.0, 'obj_loss': 0.0, 'noobj_loss': 0.0, 'cls_loss': 0.0}
        
        for prediction, label, anchor in zip(predictions, labels, config.ANCHORS):
            loss_scale, comp_loss_scale = yolo_loss.calculate_loss(jnp.array(prediction), jnp.array(label), jnp.array(anchor)) 
            total_loss += loss_scale

            for key in comp_loss_scale:
                component_loss[key] += comp_loss_scale[key]
                
        return total_loss, (states, component_loss)
    (loss_value, (states, component_loss)), gradients = (jax.value_and_grad(apply_loss, argnums=2, has_aux=True))(images, labels, params, states)

    params, optimizer_states = jax.jit(optimizer.step)(params, gradients, optimizer_states)
    return loss_value, component_loss, params, states, optimizer_states, gradients

best_loss = np.inf
for epoch in tqdm(range(20), desc="Epoch", position=0):
    if epoch == 10:
        optimizer.states['lr'] = 5e-4
    if epoch > 5:
        optimizer.states['lr'] *= 0.95
    total_losses = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = jnp.array(np.array(images))
        labels = [label.numpy() for label in labels]
        loss_value, component_loss, params, states, optimizer.states, gradients = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    
    print(f'epoch{epoch}: {[(key, value.item()) for key, value in component_loss.items()]}')

    with open(f'grad/g{epoch}', 'wb') as f:
        pickle.dump(gradients, f)
    
    if total_losses < best_loss:
        best_loss = total_losses
        with open(f'result plots/yolov3/001_best.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
            
    component_loss.update({'total_losses': total_losses, 'loss_value': loss_value, 'lr': optimizer.states['lr']})
    epoch_result.append({key: float(value) for key, value in component_loss.items()})
    if epoch % 10 == 0:
        epoch_result.to_json('summary.json')
        
        with open(f'result plots/yolov3/001_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
            
        for column_name in ['total_losses', 'loss_value', 'bbox_loss', 'obj_loss', 'noobj_loss', 'cls_loss', 'lr']:
            epoch_result.plot(column_name)
            plt.savefig('result plots/yolov3/' + column_name+'.jpg')
            plt.close()
        
with open('result plots/yolov3/001_last.pkl', 'wb') as f:
    pickle.dump((model, params, states), f)

for column_name in ['total_losses', 'loss_value', 'bbox_loss', 'obj_loss', 'noobj_loss', 'cls_loss', 'lr']:
    epoch_result.plot(column_name)
    plt.savefig(column_name+'.jpg')
    plt.close()
    
# print(type(images), type(labels))
# print(images.min(), images.max())
# print(labels.min(), labels.max())

# some_pred = jax.numpy.concatenate([labels[..., :5], labels], axis=-1)
