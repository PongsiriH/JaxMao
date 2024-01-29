import numpy as np
import cv2, os, glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from PIL import Image
import jax.nn as nn

class ImageGenerator(Sequence):
    def __init__(
        self,
        img_dir,
        num_classes,
        batch_size=32,
        image_sizes=[(416, 416)],
        change_size_interval=None,
        builder_transform=None,
    ):
        self.img_dir = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.ppm')]
        self.label_dir = [p.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.ppm', '.txt') for p in self.img_dir]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_sizes = image_sizes
        self._transform = builder_transform
        self.transform = None
        self.change_size_interval = change_size_interval
        self.image_size = 0
        self.update_transforms(0)
        
    def __len__(self):
        return int(np.ceil(len(self.img_dir) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_img_paths = self.img_dir[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_dir[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        batch_labels = []

        for img_path, label_path in zip(batch_img_paths, batch_label_paths):
            labels = read_txt_annotations(label_path, return_dict=True)
            image = np.array(Image.open(img_path).convert("RGB").resize(self.image_size))
            image = image / 255.0
                
            one_hot_labels = np.zeros((self.num_classes,), dtype=np.float32)
            for label in labels['class_labels']:
                label = int(label)
                one_hot_labels[label] = 1.0
                
            batch_images.append(image)
            batch_labels.append(one_hot_labels)

        return np.array(batch_images), np.array(batch_labels)
    
    def update_transforms(self, epoch):
        if epoch % self.change_size_interval == 0:
            i = np.random.randint(len(self.image_sizes))
            self.image_size=self.image_sizes[i]
        # if self._transform:
        #     i = np.random.randint(len(self.image_sizes))
        #     self.transform = self._transform(image_size=self.image_sizes[i])
            
    def on_epoch_end(self):
        data_pairs = list(zip(self.img_dir, self.label_dir))
        np.random.shuffle(data_pairs)
        self.img_dir, self.label_dir = zip(*data_pairs)
        
def read_txt_annotations(file_path, return_dict=False):
    annotations = {'class_labels': [], 'xywh': []} if return_dict else []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                if return_dict:
                    annotations['class_labels'].append(class_id)
                    annotations['xywh'].append([x_center, y_center, width, height])
                else:
                    annotations.append([class_id, x_center, y_center, width, height])
    return annotations


def get_gstdb():
    import albumentations as A
    transform = A.Compose([
        A.Normalize([0, 0, 0], [1, 1, 1], max_pixel_value=255.0)
    ]) 
    return ImageGenerator('/home/jaxmao/dataset/GTSDB_YOLO/images/train',
                          43,
                          32,
                          [(416, 416)],
                        #   [(416, 416), (200, 200), (500, 500), (600, 600), (700, 700), (900, 900), (1100, 1100)],
                          change_size_interval=10,
                          builder_transform=transform)

def get_imagenet(image_size=(224, 224), batch_size=8):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_loader = train_datagen.flow_from_directory(
                        '/home/jaxmao/dataset/ImageNet-mini',
                        target_size=image_size,
                        batch_size=batch_size,
                        shuffle=True,
                        class_mode='categorical'
                        )
    return train_loader

def get_mnist(image_size=None, batch_size=16):
    image_size = None
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data to fit the model and normalize
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create a data generator
    train_datagen = ImageDataGenerator()

    # Configure the data generator
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=batch_size
    )

    return train_generator
