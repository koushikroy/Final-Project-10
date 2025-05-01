import os
import random
import cv2
import numpy as np

def get_all_images_in_class(class_path):
    """
    Recursively collects all image file paths in a given class folder.

    Args:
        class_path (str): Path to the class folder.

    Returns:
        list: List of image file paths.
    """
    image_extensions = ('.jpg', '.png', '.jpeg')
    image_paths = []
    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def visualize_dataset(dataset_path, output_image="dataset_overview.png"):
    """
    Visualizes the dataset by creating an image grid with 8 rows (one for each class)
    and 10 columns (10 random images per class). Each row is labeled with the class name.

    Args:
        dataset_path (str): Path to the dataset folder organized by split and class.
        output_image (str): Name of the output image file.
    """
    # Use the 'train' split for visualization
    train_path = os.path.join(dataset_path, "train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"'train' split not found in {dataset_path}")

    # Get class names (subfolder names in 'train')
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    num_classes = len(class_names)
    assert num_classes == 8, f"The dataset must contain exactly 8 classes, but found {num_classes}."

    # Initialize the grid
    grid_rows = num_classes
    grid_cols = 10
    image_size = 224  # Resize images to 224x224 for consistency
    label_height = 50  # Height for the label area above each row
    grid_height = grid_rows * (image_size + label_height)
    grid_width = grid_cols * image_size
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background

    # Populate the grid
    for row, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)
        images = get_all_images_in_class(class_path)
        random_images = random.sample(images, min(len(images), grid_cols))  # Select up to 10 random images

        for col in range(grid_cols):
            y_start = row * (image_size + label_height) + label_height
            y_end = y_start + image_size
            x_start = col * image_size
            x_end = x_start + image_size

            if col < len(random_images):
                img_path = random_images[col]
                img = cv2.imread(img_path)
                img = cv2.resize(img, (image_size, image_size))
                grid_image[y_start:y_end, x_start:x_end] = img
            else:
                # Fill with a blank (white) image if not enough images are available
                grid_image[y_start:y_end, x_start:x_end] = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Add class label above the row
        label_position = (10, row * (image_size + label_height) + label_height // 2)
        cv2.putText(
            grid_image,
            class_name,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,  # Font scale
            (0, 0, 0),  # Black color
            2,  # Thickness
            lineType=cv2.LINE_AA
        )

    # Save the grid image
    cv2.imwrite(output_image, grid_image)
    print(f"✅ Image saved as {output_image}")

# Example usage
dataset_path = r"D:\research\deep_learning\Final-Project-10\dataset"
visualize_dataset(dataset_path)