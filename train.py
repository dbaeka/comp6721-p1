import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from utils.analysis import explore_data
from utils.dataset_loader import load_and_preprocess_images
from utils.evaluation import evaluate_model, compare_models
from utils.feature_transformation import normalize, apply_pca
from utils.model_inference import save_models, save_feature_transform
from utils.model_training import train_decision_tree, train_gradient_boosting, train_random_forest

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set random seed for reproducibility
np.random.seed(42)

if __name__ == "__main__":
    print("------------------Group Members------------------------")
    print("Delmwin Baeka (40277017)")
    print("Lordina Nkansah (40293731)")
    print("Anjolaoluwa Lasekan (40294470)")

    print("=== Indoor vs Outdoor Image Classification Training ===\n")
    print("Step 1: Loading and preprocessing images...")
    training_dir = os.path.join('data', 'train')
    training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), training_dir)

    X, y = load_and_preprocess_images(training_dir)

    print("\nStep 2: Performing exploratory data analysis...")
    explore_data(X, y, save_plots=True, output_dir='plots')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

    # Optional: Scale features
    X_train, scaler = normalize(X_train)
    X_test = scaler.transform(X_test)
    save_feature_transform(scaler, "Feature Scaler")

    # Optional: Dimensionality Reduction
    _, pca = apply_pca(X_train, X_train.shape[1])

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print("Number of components to retain 95% variance:", n_components_95)

    # Reduce dimensions to 95% variance
    X_train, pca = apply_pca(X_train, n_components_95)
    X_test = pca.transform(X_test)
    save_feature_transform(pca, "Feature PCA")

    print("\nStep 3: Training and optimizing models...")

    # Train Decision Tree
    dt_model, dt_pred = train_decision_tree(X_train, y_train, X_test, save_plots=True, output_dir='plots')
    save_models(dt_model, "Decision Tree")

    # Train Random Forest
    rf_model, rf_pred = train_random_forest(X_train, y_train, X_test, save_plots=True, output_dir='plots')
    save_models(rf_model, "Random Forest")

    # Train Gradient Boosting
    gb_model, gb_pred = train_gradient_boosting(X_train, y_train, X_test, save_plots=True, output_dir='plots')
    save_models(gb_model, "Gradient Boosting")

    print("\nStep 4: Evaluating models...")
    results = [
        evaluate_model(dt_model, X_test, y_test, dt_pred, "Decision Tree", save_plots=True, output_dir='plots'),
        evaluate_model(rf_model, X_test, y_test, rf_pred, "Random Forest", save_plots=True, output_dir='plots'),
        evaluate_model(gb_model, X_test, y_test, gb_pred, "Gradient Boosting", save_plots=True, output_dir='plots')
    ]

    # Evaluate each model

    print("\nStep 5: Comparing model performance...")
    metrics_df = compare_models(results, save_plots=True, output_dir='plots')

    print("\n=== Classification Project Complete ===")
