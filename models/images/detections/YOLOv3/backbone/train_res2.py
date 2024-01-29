import jax

from model import YOLOBackboneResidual
from dataset import get_train_generator
from jaxmao import Bind
from jaxmao import optimizers as optim
from jaxmao import losses as losses 
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score, roc_auc_score
import pickle, tqdm
import numpy as np
from utils import plot_predictions
 
key = jax.random.key(42)

train_loader = get_train_generator()

model = YOLOBackboneResidual()
params, states = model.init(key)

yolo_loss = losses.BinaryCrossEntropy()
optimizer = optim.Adam(params, 0.01)

# with Bind(model, params, states) as ctx:
#     for images, labels in train_loader:
#         break
#     pred = jax.jit(ctx.module)(images)
    
#     print(images.shape, pred.shape)
#     exit()
    
@jax.jit
def train_step(images, labels, params, states, optimizer_states):
    def apply_loss(images, labels, params, states):
        predictions, states, _ = model.apply(images, params, states)
        total_loss = yolo_loss.calculate_loss(predictions, labels) 
        return total_loss, (states)
    (loss_value, (states)), gradients = jax.value_and_grad(apply_loss, argnums=2, has_aux=True)(images, labels, params, states)
    # print(gradients)
    params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
    return loss_value, params, states, optimizer_states

print('Start trianing... ')
steps_per_epoch = len(train_loader)
best_total_losses = float('inf')
print(f'steps_per_epoch: {steps_per_epoch}')
for epoch in tqdm.tqdm(range(50), desc="epoch", position=0):
    total_losses = 0.0
    for batch_idx, (images, labels) in tqdm.tqdm(enumerate(train_loader), desc="batch", position=1, leave=False):
        loss_value, params, states, optimizer.states = train_step(images, labels, params, states, optimizer.states)
        total_losses += loss_value
    avg_loss = total_losses / len(train_loader)
    print('{}: loss: {}'.format(epoch, avg_loss))
    
    # on epoch end:
    train_loader.on_epoch_end()
    if hasattr(train_loader, 'update_transforms'):
        train_loader.update_transforms(epoch)
        
    if avg_loss < best_total_losses:
        with open(f'YOLOv3.2/backbone/results/001_best.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
            
    if (epoch+1) % 1 == 0:
        with open(f'YOLOv3.2/backbone/results/001_{epoch}.pkl', 'wb') as f:
            pickle.dump((model, params, states), f)
        
        with Bind(model, params, states) as ctx:
            predictions = ctx.module(images)
        threshold = 0.8
        binary_predictions = (predictions > threshold).astype(int)
        
        print(f'epoch {epoch}: image_shape: {images.shape}')
        # Hamming Loss
        hamming_loss_value = hamming_loss(labels, binary_predictions)
        print('\tHamming Loss: {:.2f}'.format(hamming_loss_value))

        # Subset Accuracy
        subset_accuracy = accuracy_score(labels, binary_predictions)
        print('\tSubset Accuracy: {:.2f}'.format(subset_accuracy))

        # F1 Score (Micro)
        f1_micro = f1_score(labels, binary_predictions, average='micro')
        print('\tF1 Score (Micro): {:.2f}'.format(f1_micro))

        plot_predictions(f'YOLOv3.2/backbone/001_{epoch}.jpg', images, labels, binary_predictions, class_names=np.arange(43))