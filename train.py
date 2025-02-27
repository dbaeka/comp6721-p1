import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.analysis import explore_data
from utils.evaluation import evaluate_model, compare_models
from utils.dataset_loader import load_and_preprocess_images
from utils.model_inference import save_models, save_feature_scaler
from utils.model_training import train_decision_tree

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

if __name__ == "__main__":
    print("------------------Group Members------------------------")
    print("Delmwin Baeka (40277017)")
    print("Lordina Nkansah (4029xxxx)")
    print("Anjolaoluwa Lasekan (40294470)")

    print("=== Indoor vs Outdoor Image Classification Training ===\n")
    print("Step 1: Loading and preprocessing images...")
    training_dir = 'data/train/'
    X, y = load_and_preprocess_images(training_dir)

    print("\nStep 2: Performing exploratory data analysis...")
    df = explore_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

    # Optional: Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    save_feature_scaler(scaler)

    # 4. Train and optimize models
    print("\nStep 3: Training and optimizing models...")

    # Train Decision Tree
    dt_model, dt_pred = train_decision_tree(X_train, y_train, X_test, y_test)
    save_models(dt_model, "Decision Tree")

    #
    # # Train Random Forest
    # rf_model, rf_pred = train_random_forest(X_train, y_train, X_test, y_test)
    # save_models(rf_model, "Random Forest")

    #
    # # Train Gradient Boosting
    # gb_model, gb_pred = train_gradient_boosting(X_train, y_train, X_test, y_test)
    # save_models(gb_model, "Gradient Boosting")

    # 5. Evaluate models
    print("\nStep 4: Evaluating models...")
    results = [evaluate_model(dt_model, X_test, y_test, dt_pred, "Decision Tree"),
               # evaluate_model(rf_model, X_test, y_test, rf_pred, "Random Forest"),
               # evaluate_model(gb_model, X_test, y_test, gb_pred, "Gradient Boosting")
               ]

    # Evaluate each model

    # 6. Compare models
    print("\nStep 5: Comparing model performance...")
    metrics_df = compare_models(results)

    print("\n=== Classification Project Complete ===")
