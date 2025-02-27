import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train and optimize a decision tree classifier
    """
    print("\n=== Decision Tree Classifier ===")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # Initialize the model
    dt = DecisionTreeClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
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
    plt.show()

    return best_model, y_pred


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and optimize a random forest classifier
    """
    print("\n=== Random Forest Classifier ===")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    # Initialize the model
    rf = RandomForestClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

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
    plt.show()

    return best_model, y_pred


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train and optimize a gradient boosting classifier
    """
    print("\n=== Gradient Boosting Classifier ===")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    # Initialize the model
    gb = GradientBoostingClassifier(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)

    return best_model, y_pred
