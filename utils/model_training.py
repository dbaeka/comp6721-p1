import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_decision_tree(X_train, y_train, X_test, save_plots=False, output_dir=None):
    """
    Train and optimize a decision tree classifier
    """
    print("\n=== Decision Tree Classifier ===")

    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define parameter grid for hyperparameter tuning
    param_dist = {
        'max_depth': [5, 10, 15, None],  # Limits the depth of the tree to avoid overfitting
        'min_samples_split': [2, 10, 20],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 5, 10],  # Minimum samples required to be at a leaf node
        'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for splits
        'criterion': ['gini', 'entropy'],
    }

    # Convert to float32 to reduce memory usage
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    dt = DecisionTreeClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(dt, param_dist, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Visualize the decision tree (limited to max_depth=3 for clarity)
    plt.figure(figsize=(20, 10))
    plot_tree(best_model, max_depth=3, filled=True, feature_names=None, class_names=['Indoor', 'Outdoor'])
    plt.title('Decision Tree Visualization (Limited to Depth 3)')

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'decision_tree.png'))
        plt.close()
    else:
        plt.show()

    return best_model, y_pred


def train_random_forest(X_train, y_train, X_test, save_plots=False, output_dir=None):
    """
    Train and optimize a random forest classifier
    """
    print("\n=== Random Forest Classifier ===")

    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define parameter grid for hyperparameter tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 5, 10],  # Minimum samples required to be at a leaf node
        'max_features': ['sqrt', 'log2', None],  # Number of features to consider for splits
        'criterion': ['gini', 'entropy']  # The function to measure the quality of a split
    }

    # Convert to float32 to reduce memory usage
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    rf = RandomForestClassifier(random_state=42)

    # Perform grid search with cross-validation
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                     n_iter=10, cv=5, scoring='accuracy',
                                     n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = rand_search.best_params_
    best_model = rand_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Feature importance
    feature_importance = best_model.feature_importances_
    indices = np.argsort(feature_importance)[-20:]  # Top 20 features

    plt.figure(figsize=(10, 8))
    plt.title('Random Forest Feature Importance')
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'random_forest_feature_importance.png'))
        plt.close()
    else:
        plt.show()

    return best_model, y_pred


def train_gradient_boosting(X_train, y_train, X_test, save_plots=False, output_dir=None):
    """
    Train and optimize a gradient boosting classifier
    """
    print("\n=== Gradient Boosting Classifier ===")

    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Define parameter grid for hyperparameter tuning
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.8, 1.0]
    }

    # Convert to float32 to reduce memory usage
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    gb = GradientBoostingClassifier(random_state=42)

    # Perform grid search with cross-validation
    rand_search = RandomizedSearchCV(gb, param_distributions=param_dist,
                                     n_iter=10, cv=5, scoring='accuracy',
                                     n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = rand_search.best_params_
    best_model = rand_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    # Feature importance
    feature_importance = best_model.feature_importances_
    indices = np.argsort(feature_importance)[-20:]  # Top 20 features

    plt.figure(figsize=(10, 8))
    plt.title('Gradient Boosting Feature Importance')
    plt.barh(range(len(indices)), feature_importance[indices], align='center')
    plt.yticks(range(len(indices)), [f'Feature {i}' for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'gradient_boosting_feature_importance.png'))
        plt.close()
    else:
        plt.show()

    return best_model, y_pred
