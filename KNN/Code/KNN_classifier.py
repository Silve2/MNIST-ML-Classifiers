import os
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from time import time
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle


class KNN(BaseEstimator):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.execution_times_ = []  # Store execution times separately

    def fit(self, X, y):
        self.X_train = X.astype(np.float32)
        self.y_train = np.array([str(label) for label in y])
        return self

    def predict(self, X):
        predictions = []
        t0 = time()
        for test_point in X:
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)

        execution_time = time() - t0
        self.execution_times_.append(execution_time)
        return np.array(predictions)

    def predict_proba(self, X):
        n_classes = len(np.unique(self.y_train))
        probabilities = np.zeros((X.shape[0], n_classes))

        for i, test_point in enumerate(X):
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))
            k_nearest_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            for j, class_label in enumerate(sorted(np.unique(self.y_train))):
                probabilities[i, j] = np.mean(k_nearest_labels == class_label)

        return probabilities

    def score(self, X, y):
        y = np.array([str(label) for label in y])
        predictions = self.predict(X)
        return np.mean(predictions == y)


def save_plot(fig, filename, results_dir):
    """Save a matplotlib figure to the specified directory."""
    plt.savefig(os.path.join(results_dir, "plots", filename))
    plt.close(fig)


def plot_examples_with_prob(X, y_true, y_pred, y_prob, results_dir, n_examples=5):
    """Plot example predictions with their probabilities."""
    plt.figure(figsize=(15, 3))
    for i in range(n_examples):
        plt.subplot(1, n_examples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        prob = np.max(y_prob[i])
        plt.title(f"Pred: {y_pred[i]} ({prob:.2%})\nTrue: {y_true[i]}")
        plt.axis("off")
    plt.tight_layout()
    save_plot(plt.gcf(), f"example_predictions_{i+1}.png", results_dir)


def plot_roc_curves(y_test_bin, y_score, n_classes, colors, results_dir):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(12, 8))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Class")
    plt.legend(loc="lower right")
    save_plot(plt.gcf(), "roc_curves.png", results_dir)

    return roc_auc


def main():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"mnist_knn_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "model"), exist_ok=True)
    print(f"Results directory created at: {results_dir}")

    # Load and prepare MNIST data
    print("Loading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype("float32") / 255.0

    # Take subset
    X = X[:70000]
    y = y[:70000]

    # Standard MNIST split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Feature Selection
    print("\nPerforming feature selection...")
    t0 = time()
    pre_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    pre_rf.fit(X_train, y_train)

    selector = SelectFromModel(pre_rf, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    feature_selection_time = time() - t0
    print(f"Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
    print(f"Selection time: {feature_selection_time:.2f} seconds")

    # RandomizedSearch
    param_distributions = {"k": [1, 3, 5, 7, 9, 11]}

    random_search = RandomizedSearchCV(
        estimator=KNN(),
        param_distributions=param_distributions,
        n_iter=6,
        cv=10,
        n_jobs=-1,
        verbose=2,
    )

    # Execute RandomizedSearch
    print("\nStarting RandomizedSearch...")
    t0 = time()
    random_search.fit(X_train_selected, y_train)
    search_time = time() - t0
    print(f"Search time: {search_time:.2f} seconds")
    print("Best parameters:", random_search.best_params_)
    print("Best CV score:", random_search.best_score_)

    # Evaluate final model
    best_knn = random_search.best_estimator_
    t0 = time()
    y_pred = best_knn.predict(X_test_selected)
    y_score = best_knn.predict_proba(X_test_selected)
    test_time = time() - t0

    # Convert labels for string comparison
    y_test_str = np.array([str(label) for label in y_test])
    y_pred_str = np.array([str(label) for label in y_pred])

    # Prepare for ROC and PR curves
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=sorted(np.unique(y_test)))
    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "pink",
            "brown",
        ]
    )

    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm_normalized = confusion_matrix(y_test_str, y_pred_str, normalize="true")
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    save_plot(plt.gcf(), "confusion_matrix.png", results_dir)

    # 2. ROC Curves
    roc_auc = plot_roc_curves(y_test_bin, y_score, n_classes, colors, results_dir)

    # 3. Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], y_score[:, i]
        )
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"Class {i} (AP = {average_precision[i]:.2f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves for Each Class")
    plt.legend(loc="lower left")
    save_plot(plt.gcf(), "precision_recall_curves.png", results_dir)

    # 4. Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(
        best_knn, X_train_selected, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue", marker="o")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(
        train_sizes,
        test_mean,
        label="Cross-validation score",
        color="green",
        marker="s",
    )
    plt.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.15,
        color="green",
    )
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title("Learning Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot(plt.gcf(), "learning_curves.png", results_dir)

    # 5. Example predictions with probabilities
    indices = np.random.randint(0, len(X_test), 5)
    plot_examples_with_prob(
        X_test[indices], y_test[indices], y_pred[indices], y_score[indices], results_dir
    )

    # Error Analysis
    errors_by_class = pd.DataFrame(
        {
            "true_label": y_test[y_test != y_pred],
            "predicted_label": y_pred[y_test != y_pred],
        }
    )

    error_analysis = {
        "errors_by_true_label": errors_by_class["true_label"].value_counts().to_dict(),
        "errors_by_predicted_label": errors_by_class["predicted_label"]
        .value_counts()
        .to_dict(),
    }

    # Calculate class statistics
    class_stats = {}
    for classe in sorted(np.unique(y_test)):
        mask = y_test == classe
        acc_classe = np.mean(y_pred[mask] == y_test[mask])
        n_samples = np.sum(mask)
        media_confidenza = np.mean(y_score[mask, int(classe)])
        class_stats[str(classe)] = {
            "num_samples": int(n_samples),
            "accuracy": float(acc_classe),
            "mean_confidence": float(media_confidenza),
        }

    # Create metrics dictionary
    metrics = {
        "model_parameters": {
            "best_k": best_knn.k,
            "num_features": X_train_selected.shape[1],
        },
        "training_metrics": {
            "feature_selection_time": feature_selection_time,
            "random_search_time": search_time,
            "best_cv_score": float(random_search.best_score_),
            "execution_times": best_knn.execution_times_,
        },
        "test_metrics": {
            "accuracy": float(best_knn.score(X_test_selected, y_test)),
            "inference_time": float(test_time),
            "classification_report": classification_report(y_test_str, y_pred_str),
            "class_specific_stats": class_stats,
            "average_precision_scores": {
                str(i): float(average_precision[i]) for i in range(n_classes)
            },
            "roc_auc_scores": {str(i): float(roc_auc[i]) for i in range(n_classes)},
            "error_analysis": error_analysis,
        },
    }

    # Save results
    metrics_path = os.path.join(results_dir, "metrics.json")
    print(f"\nSaving metrics to: {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save model
    model_path = os.path.join(results_dir, "model", "best_model.pkl")
    print(f"Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(best_knn, f)

    # Save feature selector
    selector_path = os.path.join(results_dir, "model", "feature_selector.pkl")
    print(f"Saving feature selector to: {selector_path}")
    with open(selector_path, "wb") as f:
        pickle.dump(selector, f)

    print(f"\nAll results have been saved in: {results_dir}")
    print("\nDirectory contents:")
    for root, dirs, files in os.walk(results_dir):
        level = root.replace(results_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


if __name__ == "__main__":
    main()
