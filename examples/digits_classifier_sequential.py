
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset
from jaxmao import Sequential, Module, Dense, Bind, BatchNorm1d
from jaxmao import losses, optimizers
import jax
from jaxmao import regularizers 
from sklearn.metrics import accuracy_score
jax.config.update('jax_check_tracer_leaks', True)
jax.config.update('JAX_TRACEBACK_FILTERING'.lower(), 'off')

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

class DenseReLu(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = Dense(in_channels, out_channels, kernel_reg=regularizers.L1(0.001))
        self.bn = BatchNorm1d(out_channels)
    def call(self, x):
        return jax.nn.relu(self.bn(self.dense(x)))    

class DenseSoftmax(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = Dense(in_channels, out_channels, kernel_reg=regularizers.L1(0.001))
        self.bn = BatchNorm1d(out_channels)
    def call(self, x):
        return jax.nn.softmax(self.bn(self.dense(x)))    
    
    
if __name__ == '__main__':
    seed = 42
    key = jax.random.key(seed=seed)
    datalodaer = get_dataloader(batch_size=32, shuffle=True)
    model = Sequential([
        DenseReLu(64, 48),
        DenseReLu(48, 32),
        DenseSoftmax(32, 10)
    ])
    
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