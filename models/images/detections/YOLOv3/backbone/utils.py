import matplotlib.pyplot as plt
import numpy as np
def plot_predictions(save_path, images, true_labels, predictions, class_names, num_images=2):
    """
    Plots the images with their predictions and true labels.

    :param images: Array of images.
    :param true_labels: Array of true labels.
    :param predictions: Array of predicted labels.
    :param class_names: List of class names.
    :param num_images: Number of images to plot.
    """
    # Select a few images randomly to display
    indices = np.random.choice(range(len(images)), num_images, replace=False)

    plt.figure(figsize=(15, num_images * 3))
    
    for i, index in enumerate(indices):
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(images[index])
        plt.title("Image {}".format(index))
        plt.xticks([])
        plt.yticks([])

        true_label_indices = np.where(true_labels[index] == 1)[0]
        pred_label_indices = np.where(predictions[index] == 1)[0]

        true_label_names = [str(class_names[idx]) for idx in true_label_indices]
        pred_label_names = [str(class_names[idx]) for idx in pred_label_indices]

        plt.xlabel("True: {}\nPredicted: {}".format(", ".join(true_label_names), ", ".join(pred_label_names)))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()