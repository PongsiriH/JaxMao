import jax
from jaxmao import Bind
from dataset import get_train_generator
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score, roc_auc_score
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_size = (240, 240)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

def get_train_loader(image_size=base_size, batch_size=32):
    train_loader = train_datagen.flow_from_directory(
                        '/home/jaxmao/dataset/ImageNet-mini',  # Replace with the path to your training data
                        target_size=image_size,
                        batch_size=batch_size,
                        shuffle=True,
                        class_mode='categorical'
                        )
    return train_loader

# train_loader = get_train_generator()
train_loader = get_train_loader()

with open('YOLOv3_2/backbone/results/001_best.pkl', 'rb') as f:
    model, params, states = pickle.load(f)

with Bind(model, params, states) as ctx:
    for images, labels in train_loader:
        break
    
    predictions = jax.jit(ctx.module)(images)
    
    print(images.shape, predictions.shape)
    print(predictions.max())
    threshold = 0.9
    binary_predictions = (predictions > threshold).astype(int)

    # Accuracy Score
    accuracy = accuracy_score(labels, binary_predictions)
    print('\tAccuracy Score: {:.2f}'.format(accuracy))
        
    # Hamming Loss
    hamming_loss_value = hamming_loss(labels, binary_predictions)
    print('\tHamming Loss: {:.2f}'.format(hamming_loss_value))

    # Subset Accuracy
    subset_accuracy = accuracy_score(labels, binary_predictions)
    print('\tSubset Accuracy: {:.2f}'.format(subset_accuracy))

    # F1 Score (Micro)
    f1_micro = f1_score(labels, binary_predictions, average='micro')
    print('\tF1 Score (Micro): {:.2f}'.format(f1_micro))