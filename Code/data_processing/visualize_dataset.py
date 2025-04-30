import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_dataset(dataset_path, output_image="dataset_overview.png"):
    """
    Visualizes the dataset by creating an image grid with 8 rows (one for each class)
    and 10 columns (10 random images per class). Each row is labeled with the class name.

    Args:
        dataset_path (str): Path to the dataset folder organized by class.
        output_image (str): Name of the output image file.
    """
    # Get class names (subfolder names)
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(class_names)
    assert num_classes == 8, "The dataset must contain exactly 8 classes."

    # Initialize the grid
    grid_rows = num_classes
    grid_cols = 10
    image_size = 224  # Resize images to 224x224 for consistency
    grid_height = grid_rows * image_size
    grid_width = grid_cols * image_size
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background

    # Populate the grid
    for row, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
        random_images = random.sample(images, min(len(images), grid_cols))  # Select up to 10 random images

        for col, img_path in enumerate(random_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (image_size, image_size))
            y_start = row * image_size
            y_end = y_start + image_size
            x_start = col * image_size
            x_end = x_start + image_size
            grid_image[y_start:y_end, x_start:x_end] = img

        # Add class label
        plt.text(5, row * image_size + image_size // 2, class_name, fontsize=10, color='black', verticalalignment='center')

    # Save the grid image
    plt.figure(figsize=(15, 12))
    plt.imshow(grid_image)
    plt.axis('off')
    plt.title("Dataset Overview")
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    print(f"✅ Image saved as {output_image}")

# Example usage
dataset_path = r"../../../dataset/train"  # Path to the dataset folder
visualize_dataset(dataset_path)