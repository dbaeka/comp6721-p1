import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

pd.set_option('display.max_columns', None)


def visualize_hyperparameters(search_results, selected_cols, model, metric='mean_test_score'):
    """
    Visualize the effect of two hyperparameters on the model performance
    """
    results_df = pd.DataFrame(search_results)
    print(f"\nHyperparameter tuning results for {model}:")
    print(results_df[selected_cols].sort_values(by=metric, ascending=False))


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

    dt = DecisionTreeClassifier(random_state=42)

    # Perform grid search with cross-validation
    rand_search = RandomizedSearchCV(dt, param_distributions=param_dist, random_state=42,
                                     cv=5, n_iter=10, n_jobs=-1, scoring='accuracy')
    rand_search.fit(X_train, y_train)

    # Evaluate the parameters used
    visualize_hyperparameters(rand_search.cv_results_,
                              selected_cols=['param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
                                             'mean_test_score', 'std_test_score'], model='Decision Tree',
                              metric='mean_test_score')

    # Get best parameters and model
    best_params = rand_search.best_params_
    best_model = rand_search.best_estimator_

    print("Best parameters:", best_params)
    print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

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

    rf = RandomForestClassifier(random_state=42)

    # Perform grid search with cross-validation
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                     n_iter=10, cv=5, scoring='accuracy',
                                     n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)

    # Evaluate the parameters used
    visualize_hyperparameters(rand_search.cv_results_,
                              selected_cols=['param_n_estimators', 'param_max_depth', 'param_min_samples_split',
                                             'param_min_samples_leaf', 'mean_test_score', 'std_test_score'],
                              model='Random Forest', metric='mean_test_score')

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

    gb = GradientBoostingClassifier(random_state=42)

    # Perform grid search with cross-validation
    rand_search = RandomizedSearchCV(gb, param_distributions=param_dist,
                                     n_iter=10, cv=5, scoring='accuracy',
                                     n_jobs=-1, random_state=42)
    rand_search.fit(X_train, y_train)

    # Evaluate the parameters used
    visualize_hyperparameters(rand_search.cv_results_,
                              selected_cols=['param_n_estimators', 'param_learning_rate', 'param_max_depth',
                                             'param_subsample', 'mean_test_score', 'std_test_score'],
                              model='Gradient Boosting', metric='mean_test_score')

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


def semi_supervised_learning(X_train, y_train, X_test, initial_label_rate=0.2,
                             confidence_threshold=0.85, iterations=5, save_plots=False, output_dir=None):
    """
       Train and optimize a semi-supervised learning model using a decision tree classifier
       """
    print("\n=== Semi-Supervised Learning w Decision Tree ===")

    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Randomly select initial labelled data (20%) and treat the rest as unlabeled
    num_train = X_train.shape[0]
    indices = np.arange(num_train)
    np.random.shuffle(indices)
    num_labelled = int(initial_label_rate * num_train)
    labelled_indices = indices[:num_labelled]
    unlabelled_indices = indices[num_labelled:]

    X_labelled = X_train[labelled_indices]
    y_labelled = np.array(y_train)[labelled_indices]
    X_unlabelled = X_train[unlabelled_indices]

    def hyperparameter_tuning(X, y):
        param_dist = {
            'max_depth': [5, 10, 15, None],  # Limits the depth of the tree to avoid overfitting
            'min_samples_split': [2, 10, 20],  # Minimum samples required to split an internal node
            'min_samples_leaf': [1, 5, 10],  # Minimum samples required to be at a leaf node
            'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for splits
            'criterion': ['gini', 'entropy'],
        }
        dt = DecisionTreeClassifier(random_state=42)
        rand_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=10, random_state=42, cv=5,
                                         scoring='accuracy', n_jobs=-1)
        rand_search.fit(X, y)

        # Evaluate the parameters used
        visualize_hyperparameters(rand_search.cv_results_,
                                  selected_cols=['param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
                                                 'mean_test_score', 'std_test_score'],
                                  model='Decision Tree Iteration',
                                  metric='mean_test_score')

        # Get best parameters and model
        best_params = rand_search.best_params_
        best_model = rand_search.best_estimator_

        print("Best parameters:", best_params)
        print(f"Best cross-validation score: {rand_search.best_score_:.4f}")

        return best_model

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        # Train the model on currently labelled data
        model = hyperparameter_tuning(X_labelled, y_labelled)

        # Predict probabilities on unlabelled data
        probas = model.predict_proba(X_unlabelled)
        # Determine maximum probabilities and predicted labels
        max_probas = np.max(probas, axis=1)
        predicted_labels = model.predict(X_unlabelled)

        # Select high-confidence predictions using threshold (e.g., >= 0.85)
        high_conf_indices = np.where(max_probas >= confidence_threshold)[0]
        print(f"Number of high-confidence pseudo-labels: {len(high_conf_indices)}")

        # If no high-confidence samples, exit the loop early
        if len(high_conf_indices) == 0:
            print("No high-confidence predictions in this iteration. Stopping.")
            break

        # Add the high-confidence pseudo-labelled samples to the labelled set
        X_labelled = np.concatenate((X_labelled, X_unlabelled[high_conf_indices]), axis=0)
        y_labelled = np.concatenate((y_labelled, predicted_labels[high_conf_indices]), axis=0)

        # Remove the pseudo-labelled samples from the unlabelled set
        X_unlabelled = np.delete(X_unlabelled, high_conf_indices, axis=0)

    # Final training on the expanded labelled set
    final_model = hyperparameter_tuning(X_labelled, y_labelled)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)

    # Visualize the decision tree (limited to max_depth=3 for clarity)
    plt.figure(figsize=(20, 10))
    plot_tree(final_model, max_depth=3, filled=True, feature_names=None, class_names=['Indoor', 'Outdoor'])
    plt.title('Decision Tree Visualization (Limited to Depth 3)')

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'semi_decision_tree.png'))
        plt.close()
    else:
        plt.show()

    return final_model, y_pred
