import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from time import time
import pandas as pd
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os
import json
import pickle
from datetime import datetime

# Definizione delle directory
base_dir = "mnist_results"  # Directory base per i risultati
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(base_dir, f"run_{timestamp}")
model_dir = os.path.join(results_dir, "model")

# Crea le directory necessarie
os.makedirs(base_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
print(f"Directory dei risultati creata in: {results_dir}")

def save_plot(fig, filename):
    fig.savefig(os.path.join(results_dir, "plots", filename))
    plt.close(fig)

# Caricamento dataset
print("Scaricamento del dataset MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0
X = X[:70000]  # Limita il dataset per una esecuzione più veloce
y = y[:70000]

# Converti le etichette in numerico
y = y.astype(np.int32)

# Split standard MNIST
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Prepara le etichette binarizzate per le curve ROC e PR
n_classes = len(np.unique(y))
y_test_bin = label_binarize(y_test, classes=range(n_classes))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'purple', 'green', 'pink', 'gray', 'brown', 'olive'])

# Feature Selection
print("\nSelezione delle feature più importanti...")
t0 = time()
pre_rf = RandomForestClassifier(n_estimators=50, random_state=42)
pre_rf.fit(X_train, y_train)

selector = SelectFromModel(pre_rf, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
feature_selection_time = time() - t0
print(f"Feature ridotte da {X_train.shape[1]} a {X_train_selected.shape[1]}")
print(f"Tempo selezione feature: {feature_selection_time:.2f} secondi")

# Definizione dei parametri per la ricerca
param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,  # Ridotto per velocizzare l'esecuzione
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# Esecuzione RandomizedSearch
print("\nInizio RandomizedSearch...")
t0 = time()
random_search.fit(X_train_selected, y_train)
search_time = time() - t0

print(f"\nTempo RandomizedSearch: {search_time:.2f} secondi")
print("Migliori parametri:", random_search.best_params_)
print("Miglior score CV:", random_search.best_score_)

# Uso del modello ottimizzato
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test_selected)
y_pred_proba = best_rf.predict_proba(X_test_selected)

# 1. Matrice di confusione normalizzata
plt.figure(figsize=(12, 10))
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
plt.title('Matrice di Confusione Normalizzata')
plt.ylabel('Label Reale')
plt.xlabel('Label Predetta')
save_plot(plt.gcf(), "confusion_matrix.png")

# 2. Precision-Recall Curve
plt.figure(figsize=(12, 8))
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Classe {i} (AP = {average_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curve Precision-Recall per ogni classe')
plt.legend(loc="lower left", bbox_to_anchor=(1.04, 0))
plt.tight_layout()
save_plot(plt.gcf(), "precision_recall_curves.png")

# 3. Learning Curves
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X_train_selected, y_train, cv=3,
    train_sizes=np.linspace(0.1, 1.0, 5))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='lower right')
plt.grid(True)
save_plot(plt.gcf(), "learning_curves.png")

# 4. Visualizzazione esempi con probabilità
def plot_examples_with_prob(X, y_true, y_pred, y_prob, n_examples=5):
    plt.figure(figsize=(15, 3))
    for i in range(n_examples):
        plt.subplot(1, n_examples, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        prob = np.max(y_prob[i])
        plt.title(f'Pred: {y_pred[i]} ({prob:.2%})\nTrue: {y_true[i]}')
        plt.axis('off')
    plt.tight_layout()
    save_plot(plt.gcf(), f"example_predictions.png")

indices = np.random.randint(0, len(X_test), 5)
plot_examples_with_prob(X_test[indices], y_test[indices], y_pred[indices], y_pred_proba[indices])

# 5. Feature importance
mask = selector.get_support()
selected_features = np.arange(len(mask))[mask]
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
})

plt.figure(figsize=(12, 6))
feature_importance.sort_values('importance', ascending=False).head(20).plot(x='feature', y='importance', kind='bar')
plt.title('Top 20 Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
save_plot(plt.gcf(), "feature_importance.png")

# 6. Analisi errori per classe
errors_by_class = pd.DataFrame({
    'true_label': y_test[y_test != y_pred],
    'predicted_label': y_pred[y_test != y_pred]
})

error_analysis = {
    'errors_by_true_label': errors_by_class['true_label'].value_counts().to_dict(),
    'errors_by_predicted_label': errors_by_class['predicted_label'].value_counts().to_dict()
}

# 7. Metriche di stabilità e performance finale
cv_results = pd.DataFrame(random_search.cv_results_)
best_score = random_search.best_score_
best_std = cv_results.loc[random_search.best_index_]['std_test_score']

# Calcola statistiche per classe
class_stats = {}
for classe in sorted(np.unique(y_test)):
    mask = y_test == classe
    acc_classe = np.mean(y_pred[mask] == y_test[mask])
    n_samples = np.sum(mask)
    media_confidenza = np.mean(y_pred_proba[mask, int(classe)])
    class_stats[str(classe)] = {
        "num_samples": int(n_samples),
        "accuracy": float(acc_classe),
        "mean_confidence": float(media_confidenza)
    }

# Crea dizionario con tutte le metriche
metrics = {
    "model_parameters": {
        "best_params": random_search.best_params_,
        "num_features": X_train_selected.shape[1]
    },
    "training_metrics": {
        "feature_selection_time": feature_selection_time,
        "random_search_time": search_time,
        "best_cv_score": float(best_score),
        "cv_score_std": float(best_std)
    },
    "feature_importance": {
        "selected_features": selected_features.tolist(),
        "importance_values": best_rf.feature_importances_.tolist()
    },
    "test_metrics": {
        "accuracy": float(best_rf.score(X_test_selected, y_test)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "class_specific_stats": class_stats,
        "average_precision_scores": {str(i): float(average_precision[i]) for i in range(n_classes)},
        "error_analysis": error_analysis
    }
}

# Salva le metriche in formato JSON
metrics_path = os.path.join(results_dir, "metrics.json")
print(f"\nSalvataggio metriche in: {metrics_path}")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Salva il modello
model_path = os.path.join(model_dir, "best_model.pkl")
print(f"Salvataggio modello in: {model_path}")
with open(model_path, "wb") as f:
    pickle.dump(best_rf, f)

# Salva il selettore di features
selector_path = os.path.join(model_dir, "feature_selector.pkl")
print(f"Salvataggio feature selector in: {selector_path}")
with open(selector_path, "wb") as f:
    pickle.dump(selector, f)

# Salva i risultati della CV come CSV
cv_path = os.path.join(results_dir, "cv_results.csv")
print(f"Salvataggio risultati CV in: {cv_path}")
cv_results.to_csv(cv_path, index=False)

print("\nContenuto della cartella dei risultati:")
for root, dirs, files in os.walk(results_dir):
    level = root.replace(results_dir, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")