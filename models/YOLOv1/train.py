import jax

from dataset import YOLOv1DataLoader
from loss import YOLO_2bboxes
from model import FullyConvolutionalYOLOv1
from jaxmao.modules import Summary
import jaxmao.optimizers as optim

import pickle

key = jax.random.key(42)

train_loader = YOLOv1DataLoader(
                    image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train',
                    label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train',
                    image_size=(490, 490),
                    num_grids=13,
                    num_classes=4,
                    batch_size=16,
                    normalize=True,
                    augment=None,
                    image_format='.jpg'
                )

model = FullyConvolutionalYOLOv1()
params, states = model.init(key)
# with open('model4.pkl', 'rb') as f:
#     model, params, states = pickle.load(f)
    
yolo_loss = YOLO_2bboxes()
optimizer = optim.Adam(params, 0.0001)

@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss, component_loss = yolo_loss.calculate_loss(predictions, labels) 
        return total_loss, (states, component_loss)
    (loss_value, (states, component_loss)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=True)(images, labels, params, states)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, component_loss, params, states, optimizer_states

with Summary(model) as ctx:
    ctx.summary((16, 490, 490, 3))

for epoch in range(2000):
    if epoch == 5:
        optimizer.states['lr'] = 0.001
    if epoch > 5:
        optimizer.states['lr'] *= 0.99
    total_losses = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        loss_value, component_loss, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    print('{}: loss: {}, last: {} {}'.format(epoch, total_losses/len(train_loader), loss_value, component_loss))

    if epoch % 200 == 0:
        with open(f'model6_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
            
with open('model6_las6.pkl', 'wb') as f:
    pickle.dump((model, params, states), f)

# print(type(images), type(labels))
# print(images.min(), images.max())
# print(labels.min(), labels.max())

# some_pred = jax.numpy.concatenate([labels[..., :5], labels], axis=-1)
