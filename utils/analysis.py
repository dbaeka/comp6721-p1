import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def explore_data(X, y, feature_names=None):
    """
    Perform exploratory data analysis on extracted features
    """
    if feature_names is None:
        # Create generic feature names if not provided
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y

    # Basic statistics
    print("Dataset shape:", X.shape)
    print(f"Number of indoor images: {np.sum(y == 0)}")
    print(f"Number of outdoor images: {np.sum(y == 1)}")

    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df)
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Indoor, 1: Outdoor)')
    plt.ylabel('Count')
    plt.show()

    # Feature analysis
    # Select a subset of features for visualization
    selected_features = feature_names[:5]  # First 5 features

    # Box plots for selected features by class
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(selected_features):
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='class', y=feature, data=df)
        plt.title(f'{feature} by Class')
    plt.tight_layout()
    plt.show()

    # Feature correlation
    plt.figure(figsize=(10, 8))
    selected_df = df[selected_features + ['class']]
    correlation = selected_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

    # Feature importance based on correlation with the target
    corr_with_target = df.corr()['class'].sort_values(ascending=False)
    print("\nTop 10 features correlated with target:")
    print(corr_with_target[:10])

    return df
