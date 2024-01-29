import jax
from dataset import get_gstdb, get_imagenet, get_mnist
from model import YOLOBackboneResidual
from jaxmao import Bind
from jaxmao import optimizers as optim
from jaxmao import losses as losses 
from sklearn.metrics import accuracy_score
import pickle, tqdm
import numpy as np
import random

key = jax.random.key(5)
image_sizes = [(224, 224), (320, 320), (352, 352), (384, 384), 
               (416, 416), (448, 448), (480, 480), (512, 512),
               (544, 544), (576, 576), (608, 608), (640, 640)]
batch_sizes = [32, 32, 32, 32,
               16, 16, 16, 16,
               4, 4, 4, 4]
assert len(image_sizes) == len(batch_sizes)
# image_sizes = [(224, 224)]
# batch_sizes = [32]
# image_sizes = [(50, 50), (100, 100), (200, 200), (360, 360), (416, 416)]
# image_sizes = [(240, 240), (300, 300), (360, 360), (416, 416), (500, 500), (600, 600)]
base_size = image_sizes[0]

train_loaders = [get_imagenet(image_size, batch_size) for image_size, batch_size in zip(image_sizes, batch_sizes)]
test_loaders = train_loaders.copy()

def init_model(key):
    model = YOLOBackboneResidual(num_classes=100)
    params, states = model.init(key)
    return model,params,states

def load_model(path):
    with open(path, 'rb') as f:
        model, params, states = pickle.load(f)
    return model,params,states

model: YOLOBackboneResidual
# model, params, states = init_model(key)
model, params, states = load_model('YOLOv3_3/backbone/results/yolov3_best.pkl')
params, states = model.remove_clf_head(params, states)
del model.clf_head
model.clf_head = model._build_clf_head()
model.submodules['clf_head'] = model.clf_head
params['clf_head'], states['clf_head'] = model.clf_head.init(jax.random.split(key, 2)[1])

loss_fn = losses.CCEWithLogitsLoss()
optimizer = optim.Adam(params, 0.05)

@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss = loss_fn.calculate_loss(predictions, labels) 
        return total_loss, (states)
    (loss_value, (states)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=True)(images, labels, params, states)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, params, states, optimizer_states

print('Start trianing... ')
# steps_per_epoch = len(train_loader)
best_avg_acc = -1
for epoch in tqdm.tqdm(range(5000), desc="epoch", position=0):
    jidx = random.randint(0, len(train_loaders))
    train_loader = train_loaders[jidx]
    total_losses = 0.0
    steps_per_epoch = int(32 / batch_sizes[jidx] * 200)
    print(f'steps_per_epoch: {steps_per_epoch}')
    for batch_idx in tqdm.tqdm(range(steps_per_epoch), desc="batch", position=1, leave=False):
        (images, labels) = next(train_loader)
        loss_value, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    avg_loss = total_losses / len(train_loader)
    print('{}: loss: {}'.format(epoch, avg_loss))
    
    optimizer.states['lr'] *= 0.95

    number_of_batches = len(train_loaders) * 5
    with Bind(model, params, states) as ctx:
        accuracies = []
        for image_size, test_loader in zip(image_sizes, test_loaders):
            scale_accuracies = []
            for i, (images, labels) in enumerate(test_loader): 
                if i >= number_of_batches:  # limit the number of batches to predict
                    break
                predictions = jax.jit(ctx.module)(images)
                batch_accuracy = accuracy_score(labels.argmax(axis=1), predictions.argmax(axis=1))
                scale_accuracies.append(batch_accuracy)
            average_accuracy = np.mean(scale_accuracies)
            accuracies.append(average_accuracy)
            print('image_size{}: average_accuracy{:.2f}'.format(image_size, average_accuracy))
            
    avg_accuracies = float(np.mean(accuracies))
    if avg_accuracies < best_avg_acc:
        with open(f'YOLOv3_3/backbone/results/yolov3_replace_head_best.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
            
    if (epoch+1) % 20 == 0:
        with open(f'YOLOv3_3/backbone/results/yolov3_replace_head_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)