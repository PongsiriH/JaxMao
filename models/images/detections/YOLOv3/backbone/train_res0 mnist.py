import jax
from dataset import get_gstdb, get_imagenet, get_mnist
from model import YOLOBackboneResidual
from jaxmao import Bind
from jaxmao import optimizers as optim
from jaxmao import losses as losses 
from sklearn.metrics import accuracy_score
import pickle, tqdm
import numpy as np

key = jax.random.key(42)

train_loader = get_mnist()

model = YOLOBackboneResidual(in_channels=1, num_classes=10)
params, states = model.init(key)

yolo_loss = losses.CCEWithLogitsLoss()
optimizer = optim.Adam(params, 0.1)

@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss = yolo_loss.calculate_loss(predictions, labels) 
        return total_loss, (states)
    (loss_value, (states)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=True)(images, labels, params, states)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, params, states, optimizer_states

print('Start trianing... ')
steps_per_epoch = len(train_loader)
best_total_losses = float('inf')
print(f'steps_per_epoch: {steps_per_epoch}')
for epoch in tqdm.tqdm(range(5000), desc="epoch", position=0):
    total_losses = 0.0
    for batch_idx in tqdm.tqdm(range(steps_per_epoch), desc="batch", position=1, leave=False):
        (images, labels) = next(train_loader)
        loss_value, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    avg_loss = total_losses / len(train_loader)
    print('51')
    print('{}: loss: {}'.format(epoch, avg_loss))
    
    optimizer.states['lr'] *= 0.99
    
    number_of_batches = 10
    with Bind(model, params, states) as ctx:
        accuracies = []
        for i, (images, labels) in enumerate(train_loader): 
            if i >= number_of_batches:  # limit the number of batches to predict
                break
            predictions = ctx.module(images)
            batch_accuracy = accuracy_score(labels.argmax(axis=1), predictions.argmax(axis=1))
            accuracies.append(batch_accuracy)

            average_accuracy = sum(accuracies) / len(accuracies)
            print('image_size (28, 28): average_accuracy{:.2f}'.format(average_accuracy))

    if (epoch+1) % 5 == 0:
        with open(f'YOLOv3_2/backbone/results/003_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)