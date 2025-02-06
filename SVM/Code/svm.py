from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
import os
from datetime import datetime

# Caricamento dataset
print("Caricamento dataset...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
y = y.astype(int)
X = X / 255.0

# Split train/test come specificato nel dataset originale
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Definizione dei param_grid per ogni kernel
param_grids = {
    "linear": {"C": [0.1, 1, 10]},
    "poly": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "degree": [2]},
    "rbf": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
}

# 10-fold cross validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Dizionario per salvare i risultati
models_results = {}

# Creazione cartella per i risultati
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"mnist_svm_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Per ogni kernel, addestra un modello separato
for kernel_name, param_grid in param_grids.items():
    print(f"\nAddestramento modello con kernel {kernel_name}...")

    # Creazione del modello base
    base_model = SVC(kernel=kernel_name, random_state=42)

    # Grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        refit="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    # Training e misurazione tempo
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Test set evaluation
    start_time = time.time()
    y_pred = grid_search.predict(X_test)
    predict_time = time.time() - start_time

    # Salvataggio risultati
    models_results[kernel_name] = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "train_time": train_time,
        "predict_time": predict_time,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "cv_results": pd.DataFrame(grid_search.cv_results_),
    }

    # Salvataggio del modello
    model_path = os.path.join(results_dir, f"svm_{kernel_name}_model.joblib")
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Modello {kernel_name} salvato in {model_path}")

    # Plot della matrice di confusione
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        models_results[kernel_name]["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title(f"Matrice di Confusione - Kernel {kernel_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{kernel_name}.png"))
    plt.close()

# Salvataggio dei risultati in un file di testo
results_path = os.path.join(results_dir, "results.txt")
with open(results_path, "w") as f:
    f.write("=== Risultati SVM su MNIST ===\n\n")
    f.write(f"Data e ora: {timestamp}\n\n")

    for kernel_name, results in models_results.items():
        f.write(f"\n=== Modello {kernel_name} ===\n")
        f.write("Migliori parametri:\n")
        for param, value in results["best_params"].items():
            f.write(f"{param}: {value}\n")
        f.write(
            f"\nMiglior accuracy in cross-validation: {results['best_score']:.4f}\n"
        )
        f.write(f"Tempo di training: {results['train_time']:.2f} secondi\n")
        f.write(f"Tempo di predizione: {results['predict_time']:.2f} secondi\n\n")
        f.write("Report di classificazione:\n")
        f.write(results["classification_report"])
        f.write("\n" + "=" * 50 + "\n")

# Salvataggio dei risultati della grid search per ogni modello
for kernel_name, results in models_results.items():
    results["cv_results"].to_csv(
        os.path.join(results_dir, f"grid_search_results_{kernel_name}.csv")
    )

print(f"\nTutti i risultati sono stati salvati nella cartella: {results_dir}")

# Esempio di come caricare i modelli
print("\nPer caricare i modelli in futuro, usa il seguente codice:")
for kernel_name in param_grids.keys():
    print(
        f"model_{kernel_name} = joblib.load('{results_dir}/svm_{kernel_name}_model.joblib')"
    )
