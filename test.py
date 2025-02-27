import warnings

import numpy as np

from utils.dataset_loader import load_and_preprocess_images
from utils.evaluation import evaluate_model, compare_models
from utils.model_inference import load_model, load_feature_transform

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

if __name__ == "__main__":
    print("------------------Group Members------------------------")
    print("Delmwin Baeka (40277017)")
    print("Lordina Nkansah (4029xxxx)")
    print("Anjolaoluwa Lasekan (40294470)")

    print("=== Indoor vs Outdoor Image Classification Testing ===\n")
    print("Step 1: Loading and preprocessing images...")

    # 1. Load the models
    dt_model = load_model("Decision Tree 2")
    # rf_model, _ = load_model("Random Forest")
    # gb_model = load_model("Gradient Boosting")

    # Load the feature scaler
    feature_scaler = load_feature_transform("Feature Scaler 2")

    # Load the PCA transform
    pca_transform = load_feature_transform("Feature PCA")

    # 2. Load and preprocess the test data
    print("Loading and preprocessing test images...")
    testing_dir = 'data/test/'
    X_test, y_test = load_and_preprocess_images(testing_dir)

    # Apply scaling if the model was trained on scaled data
    if feature_scaler:
        X_test = feature_scaler.transform(X_test)

    if pca_transform:
        X_test = pca_transform.transform(X_test)

    # 3. Make predictions with each model
    print("Making predictions...")
    dt_pred = dt_model.predict(X_test)
    # rf_pred = rf_model.predict(X_test)
    # gb_pred = gb_model.predict(X_test)

    # 4. Evaluate each model on the test set
    results = [evaluate_model(dt_model, X_test, y_test, dt_pred, "Decision Tree"),
               # evaluate_model(rf_model, X_test, y_test, rf_pred, "Random Forest"),
               # evaluate_model(gb_model, X_test, y_test, gb_pred, "Gradient Boosting")
               ]

    # 5. Compare models
    metrics_df = compare_models(results)

    print("\n=== Classification Project Complete ===")
