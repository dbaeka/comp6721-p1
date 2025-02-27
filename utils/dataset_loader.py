import multiprocessing as mp
import os
from functools import partial

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm


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


def process_image(img_file, class_dir, target_size, label):
    """
    Process a single image - this function will be used by the parallel workers
    """
    img_path = os.path.join(class_dir, img_file)
    try:
        # Load and resize image
        img = Image.open(img_path)
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img)

        # Extract features
        feature_vector = extract_features(img_array)

        return feature_vector, label, None
    except Exception as e:
        return None, None, f"Error processing {img_path}: {e}"


def load_and_preprocess_images(data_dir, target_size=(128, 128), max_images=None, n_jobs=None):
    """
    Load images from directory structure and extract features.
    Uses parallel processing to speed up feature extraction.
    """
    if n_jobs is None:
        # Use all available cores by default
        n_jobs = mp.cpu_count()

    features = []
    labels = []
    classes = os.listdir(data_dir)

    label_map = {'indoor': 0, 'outdoor': 1}

    # Configure a process pool for parallel processing
    pool = mp.Pool(processes=n_jobs)

    for class_name in classes:
        if class_name not in label_map:
            continue

        class_dir = os.path.join(data_dir, class_name)
        label = label_map[class_name]
        image_files = os.listdir(class_dir)

        if max_images and len(image_files) > max_images:
            image_files = np.random.choice(image_files, max_images, replace=False)

        print(f"\nProcessing {len(image_files)} {class_name} images using {n_jobs} processes...")

        # Create a partial function with fixed parameters
        process_func = partial(process_image, class_dir=class_dir, target_size=target_size, label=label)

        # Map the function to all images in parallel
        results = list(tqdm(
            pool.imap(process_func, image_files),
            total=len(image_files)
        ))

        # Process results
        for feature_vector, label_value, error in results:
            if error:
                print(error)
            elif feature_vector is not None:
                features.append(feature_vector)
                labels.append(label_value)

    pool.close()
    pool.join()

    return np.array(features), np.array(labels)
