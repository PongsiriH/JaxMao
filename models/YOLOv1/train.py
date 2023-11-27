import jax

from dataset import YOLOv1DataLoader
from loss import YOLO_2bboxes
from model import YOLOv1
from jaxmao.modules import Bind, Summary
import jaxmao.optimizers as optim

key = jax.random.key(42)

train_loader = YOLOv1DataLoader(
                    image_dir='/home/jaxmao/dataset/GTSDB_YOLO/images/train',
                    label_dir='/home/jaxmao/dataset/GTSDB_YOLO/labels/train',
                    image_size=(224, 224),
                    num_grids=7,
                    num_classes=4,
                    batch_size=10,
                    normalize=True,
                    augment=None,
                    image_format='.jpg'
                )

model = YOLOv1()
params, states = model.init(key)
yolo_loss = YOLO_2bboxes()
optimizer = optim.Adam(params, 0.001)
# print(params.keys())
# print(params['conv'].keys())
# print(params['fc'].keys())
# print(jax.tree_util.tree_structure(params))

# @jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss, component_loss = yolo_loss.calculate_loss(predictions, labels) 
        return total_loss, (states, component_loss)
    (loss_value, (states, component_loss)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=TimeoutError)(images, labels, params, states)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, component_loss, params, states, optimizer_states

with Summary(model) as ctx:
    ctx.summary((10, 224, 224, 3))
    
for batch_idx, (images, labels) in enumerate(train_loader):
    loss_value, component_loss, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
    print('{}: loss: {}, {}'.format(batch_idx, loss_value, component_loss))
    break
    
# print(type(images), type(labels))
# print(images.min(), images.max())
# print(labels.min(), labels.max())

# some_pred = jax.numpy.concatenate([labels[..., :5], labels], axis=-1)
