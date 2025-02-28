import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def create_class_distribution_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='class', data=df, ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class (0: Indoor, 1: Outdoor)')
    ax.set_ylabel('Count')
    return fig


def create_feature_boxplot(df, feature, ax):
    sns.boxplot(x='class', y=feature, data=df, ax=ax)
    ax.set_title(f'{feature} by Class')
    return ax


def create_correlation_heatmap(selected_df):
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = selected_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    return fig


def explore_data(X, y, feature_names=None, save_plots=False, output_dir=None):
    """
    Perform exploratory data analysis on extracted features with parallel processing
    """
    if feature_names is None:
        # Create generic feature names if not provided
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y

    # Basic statistics - fast operations, no need to parallelize
    print("Dataset shape:", X.shape)
    print(f"Number of indoor images: {np.sum(y == 0)}")
    print(f"Number of outdoor images: {np.sum(y == 1)}")

    # Create output directory if saving plots
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Class distribution plot
    class_dist_fig = create_class_distribution_plot(df)
    if save_plots:
        class_dist_fig.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close(class_dist_fig)
    else:
        plt.figure(class_dist_fig.number)
        plt.show()

    # Select a subset of features for visualization
    # Optimize by analyzing only the most informative features
    # We can calculate variance and select features with highest variance
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-5:]  # Top 5 features by variance
    selected_features = [feature_names[i] for i in top_indices]

    # Box plots for selected features by class
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        if i < len(axes):
            create_feature_boxplot(df, feature, axes[i])

    plt.tight_layout()
    if save_plots:
        fig.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
        plt.close(fig)
    else:
        plt.figure(fig.number)
        plt.show()

    # Feature correlation - optimized to use only selected features
    # This significantly reduces computation for large feature sets
    selected_df = df[selected_features + ['class']]
    corr_fig = create_correlation_heatmap(selected_df)

    if save_plots:
        corr_fig.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close(corr_fig)
    else:
        plt.figure(corr_fig.number)
        plt.show()

    # Feature importance based on correlation with the target
    # Optimize by calculating correlation only for the most important features
    # first by using a faster metric like mutual information
    from sklearn.feature_selection import mutual_info_classif

    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)
    mi_indices = np.argsort(mi_scores)[-50:]  # Top 50 features by MI

    # Only calculate correlation for these features
    important_features = [feature_names[i] for i in mi_indices]
    important_df = df[important_features + ['class']]

    # Calculate correlation with target (faster with subset)
    corr_with_target = important_df.corr()['class'].sort_values(ascending=False)
    print("\nTop 10 features correlated with target:")
    print(corr_with_target[:10])


def visualize_data(training_dir, n_samples=5, save_plots=False, output_dir=None):
    # Create output directory if saving plots
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 8))

    # Get outdoor images
    outdoor_dir = os.path.join(training_dir, 'outdoor')
    outdoor_files = [os.path.join(outdoor_dir, f) for f in os.listdir(outdoor_dir)
                     if os.path.isfile(os.path.join(outdoor_dir, f))]
    outdoor_samples = random.sample(outdoor_files, n_samples)

    # Get indoor images
    indoor_dir = os.path.join(training_dir, 'indoor')
    indoor_files = [os.path.join(indoor_dir, f) for f in os.listdir(indoor_dir)
                    if os.path.isfile(os.path.join(indoor_dir, f))]
    indoor_samples = random.sample(indoor_files, n_samples)

    for i, img_path in enumerate(outdoor_samples):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Outdoor {i + 1}")
        plt.axis('off')

    for i, img_path in enumerate(indoor_samples):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Indoor {i + 1}")
        plt.axis('off')

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'sample_images.png'))
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
