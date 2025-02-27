import os

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm


def load_and_preprocess_images(data_dir, target_size=(128, 128), max_images=None):
    """
    Load images from directory structure and extract features.
    """
    features = []
    labels = []
    classes = os.listdir(data_dir)

    # Create a label mapping dictionary
    label_map = {'indoor': 0, 'outdoor': 1}

    for class_name in classes:
        if class_name not in label_map:
            continue

        class_dir = os.path.join(data_dir, class_name)
        label = label_map[class_name]
        image_files = os.listdir(class_dir)

        if max_images and len(image_files) > max_images:
            image_files = np.random.choice(image_files, max_images, replace=False)

        print(f"\nProcessing {len(image_files)} {class_name} images...")

        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load and resize image
                img = Image.open(img_path)
                img = img.resize(target_size)
                img = img.convert('RGB')
                img_array = np.array(img)

                # Extract features
                feature_vector = extract_features(img_array)

                features.append(feature_vector)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(features), np.array(labels)


def extract_features(img_array):
    """
    Extract multiple features from an image
    """
    features = []

    # 1. Color histograms (RGB)
    for i in range(3):
        channel_hist, _ = np.histogram(img_array[:, :, i], bins=32, range=(0, 256))
        features.extend(channel_hist)

    # 2. Convert to grayscale for HOG features
    gray_img = rgb2gray(img_array)

    # 3. HOG features (Shape and Texture)
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)
    features.extend(hog_features)

    # 4. Local Binary Patterns (Texture)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    features.extend(lbp_hist)

    # 5. Basic statistics (brightness, contrast)
    for i in range(3):
        features.append(np.mean(img_array[:, :, i]))  # Mean intensity
        features.append(np.std(img_array[:, :, i]))  # Standard deviation (contrast)

    return np.array(features)
