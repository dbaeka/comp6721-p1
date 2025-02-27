import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve)


def evaluate_model(model, X_test, y_test, y_pred, model_name):
    """
    Evaluate model performance with multiple metrics
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print(f"\n--- {model_name} Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Indoor', 'Outdoor'])
    plt.yticks([0.5, 1.5], ['Indoor', 'Outdoor'])
    plt.tight_layout()
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Indoor', 'Outdoor']))

    # ROC Curve and AUC
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            plt.show()

            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{model_name} Precision-Recall Curve')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.show()
        except:
            print("Could not generate probability-based plots")

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compare_models(results):
    """
    Compare performance of different models
    """
    # Prepare data for plotting
    models = [result['model_name'] for result in results]
    accuracy = [result['accuracy'] for result in results]
    precision = [result['precision'] for result in results]
    recall = [result['recall'] for result in results]
    f1 = [result['f1'] for result in results]

    # Bar chart comparing metrics
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }, index=models)

    plt.figure(figsize=(12, 8))
    metrics_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Create a summary table
    print("\n=== Model Performance Summary ===")
    print(metrics_df)

    # Find the best model based on F1 score
    best_model_idx = np.argmax(f1)
    print(f"\nBest model based on F1 score: {models[best_model_idx]}")

    return metrics_df
