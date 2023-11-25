
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset
from jaxmaov2.modules import Module, Dense, Bind
from jaxmaov2 import losses, optimizers
import jax
from jaxmaov2 import regularizers 
from sklearn.metrics import accuracy_score


def get_dataloader(batch_size, shuffle):
    class DigitsDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
            
        def __len__(self):
            return len(self.data)
    
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
        
    digits = load_digits()
    data = digits.data
    targets = digits.target
    
    dataset = DigitsDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class MnistClassifier(Module):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64, 48, kernel_reg=regularizers.L1(0.001))
        self.dense2 = Dense(48, 32, kernel_reg=regularizers.L1(0.001))
        self.dense3 = Dense(32, 10)
        
    def call(self, x):
        x = jax.nn.relu(self.dense1(x))
        x = jax.nn.relu(self.dense2(x))
        x = jax.nn.softmax(self.dense3(x))
        return x
    
if __name__ == '__main__':
    seed = 42
    key = jax.random.key(seed=seed)
    datalodaer = get_dataloader(batch_size=32, shuffle=True)
    model = MnistClassifier()
    params, states = model.init(key)
    optimizer = optimizers.GradientDescent(params=params, lr=0.01)
    loss_fn = losses.CategoricalCrossEntropy()
    
    @jax.jit
    def train_step(images, targets, params, states, optimizer_states):
        def loss(images, params, states):
            predictions, states, reg = model.apply(images, params, states)
            return loss_fn(predictions, targets) + reg, states

        (loss_value, states), gradients = jax.value_and_grad(loss, argnums=1, has_aux=True)(images, params, states)
        params, optimizer_states = optimizer.step(params, gradients, optimizer_states)
        return loss_value, params, states, optimizer_states
    
    for epoch in range(20):
        losses_value = 0
        for i, (images, targets) in enumerate(datalodaer):
            images = jax.device_put(images.numpy())
            targets = jax.nn.one_hot(jax.device_put(targets.numpy()), num_classes=10)
            loss_value, params, states, optimizer.states = train_step(images, targets, params, states, optimizer.states)
            losses_value += loss_value
        print('losses_avg: ', losses_value / len(datalodaer))

    with Bind(model, params, states) as ctx:
        for i, (images, batch_targets) in enumerate(datalodaer): 
            images = jax.device_put(images.numpy())
            batch_predictions = ctx.module(images)
            accuracy = accuracy_score(batch_targets, batch_predictions.argmax(axis=1))
            print('batch {} accuracy: {}'.format(i, accuracy))