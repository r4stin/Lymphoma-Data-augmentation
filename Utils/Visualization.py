"""
    This file contains the functions to visualize the data samples and the results of the model.
"""
import matplotlib.pyplot as plt


def app_sample_visualizer(train_loader):
    num_images = 8
    images_per_row = 4

    for inputs, labels in train_loader:
        num_rows = (num_images + images_per_row - 1) / images_per_row
        num_rows = int(num_rows)

        plt.figure(figsize=(images_per_row * 5, num_rows * 5))

        for idx in range(num_images):
            plt.subplot(num_rows, images_per_row, idx + 1)
            plt.imshow(inputs[idx].permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.title(f"Label: {labels[idx]}")

        plt.tight_layout()
        plt.show()
        break


def results_visualizer(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()