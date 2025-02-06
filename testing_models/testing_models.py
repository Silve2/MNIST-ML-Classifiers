import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from joblib import load
import pickle
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
from typing import Dict, List, Tuple
from scipy.special import betaln
from sklearn.base import BaseEstimator
from time import time


# Definizione della classe BetaNaiveBayes
class BetaNaiveBayes:
    def __init__(self, smooth_factor=1.0, feature_threshold=0.1):
        self.classes_ = None
        self.alpha_ = None
        self.beta_ = None
        self.class_priors_ = None
        self.smooth_factor = smooth_factor
        self.feature_threshold = feature_threshold

    def _beta_log_pdf(self, x, alpha, beta):
        """
        Log-pdf con stabilità numerica migliorata
        """
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return (alpha - 1) * np.log(x) + (beta - 1) * np.log1p(-x) - betaln(alpha, beta)

    def predict_proba(self, X):
        X = self._preprocess_data(X)
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes_)))

        for i in range(len(self.classes_)):
            log_probs[:, i] = np.log(self.class_priors_[i])
            feature_ll = self._beta_log_pdf(X, self.alpha_[i], self.beta_[i])
            log_probs[:, i] += np.sum(feature_ll, axis=1) / X.shape[1]

        log_probs -= np.max(log_probs, axis=1)[:, np.newaxis]
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1)[:, np.newaxis]
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _preprocess_data(self, X):
        X_thresh = np.where(X > self.feature_threshold, X, 0)
        X_norm = X_thresh / (np.max(X_thresh, axis=0) + 1e-6)
        eps = 1e-6
        return np.clip(X_norm, eps, 1 - eps)


# Definizione della classe KNN
class KNN(BaseEstimator):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.execution_times_ = []

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


# Funzioni di utility
def load_all_models(model_dir: str) -> Dict[str, object]:
    """
    Carica tutti i modelli presenti nella directory specificata
    """
    print(f"Cerco modelli nella directory: {model_dir}")
    models = {}
    print("File trovati nella directory:")

    # Carica prima il feature selector se presente
    feature_selector = None
    for filename in os.listdir(model_dir):
        if filename == "feature_selector.pkl":
            with open(os.path.join(model_dir, filename), "rb") as f:
                feature_selector = pickle.load(f)
                print("Feature selector caricato con successo")

    # Carica i modelli
    for filename in os.listdir(model_dir):
        if filename == "feature_selector.pkl":
            continue  # Salta il feature selector

        print(f"- {filename}")
        filepath = os.path.join(model_dir, filename)
        try:
            if filename.endswith(".joblib"):
                model = load(filepath)
                models[filename] = model
                print(f"Modello {filename} caricato con successo")

            elif filename.endswith((".pkl")):
                with open(filepath, "rb") as file:
                    model = pickle.load(file)
                models[filename] = model
                print(f"Modello {filename} caricato con successo")

        except Exception as e:
            print(f"Errore nel caricamento del modello {filename}: {str(e)}")

    return models, feature_selector


def load_mnist_test_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica un sottoinsieme del dataset MNIST per il test
    """
    print("Caricamento dati MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    # Normalizza i dati
    X = X / 255.0

    # Converti le etichette in interi
    y = y.astype(int)

    # Prendi un campione casuale per il test
    indices = random.sample(range(len(X)), n_samples)
    X_test = X[indices]
    y_test = y[indices]

    return X_test, y_test


def preprocess_data(X: np.ndarray, model_name: str, model_dir: str) -> np.ndarray:
    """
    Preprocessa i dati in base al modello
    """
    if "random_forest" in model_name.lower():
        feature_selector_path = os.path.join(model_dir, "feature_selector.pkl")
        if os.path.exists(feature_selector_path):
            with open(feature_selector_path, "rb") as f:
                selector = pickle.load(f)
            X = selector.transform(X)
            print(
                f"Applicate feature selection: da {X.shape[1]} a {X.shape[1]} features"
            )
    return X


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    model_dir: str,
) -> Tuple[float, np.ndarray]:
    """
    Valuta le performance del modello
    """
    # Preprocessa i dati se necessario
    X_test_processed = preprocess_data(X_test, model_name, model_dir)

    # Fai le predizioni
    y_pred = model.predict(X_test_processed)

    # Assicurati che le predizioni siano dello stesso tipo delle etichette di test
    y_pred = y_pred.astype(int)
    y_test = y_test.astype(int)

    # Calcola l'accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nRisultati per {model_name}:")
    print(f"Accuratezza: {accuracy:.4f}")

    try:
        # Stampa il report di classificazione
        print("\nReport di classificazione:")
        print(classification_report(y_test, y_pred))

        # Crea e visualizza la matrice di confusione
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matrice di confusione - {model_name}")
        plt.ylabel("Label Reale")
        plt.xlabel("Label Predetta")
        plt.show()
    except Exception as e:
        print(f"Errore nella generazione dei report per {model_name}: {str(e)}")

    return accuracy, y_pred


def visualize_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_samples: int = 5,
):
    """
    Visualizza alcune predizioni di esempio per tutti i modelli
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(
        n_samples, n_models + 1, figsize=(4 * (n_models + 1), 2 * n_samples)
    )

    indices = random.sample(range(len(X_test)), n_samples)

    for i, idx in enumerate(indices):
        # Immagine originale
        axes[i, 0].imshow(X_test[idx].reshape(28, 28), cmap="gray")
        axes[i, 0].set_title(f"Reale: {y_test[idx]}")
        axes[i, 0].axis("off")

        # Predizioni di ogni modello
        for j, (model_name, y_pred) in enumerate(predictions.items(), 1):
            axes[i, j].imshow(X_test[idx].reshape(28, 28), cmap="gray")
            axes[i, j].set_title(f"{model_name}:\n{y_pred[idx]}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(accuracies: Dict[str, float]):
    """
    Crea un grafico a barre per confrontare le accuratezze dei modelli
    """
    plt.figure(figsize=(12, 6))
    models = list(accuracies.keys())
    accs = list(accuracies.values())

    bars = plt.bar(models, accs)
    plt.title("Confronto Accuratezza tra i Modelli")
    plt.xlabel("Modello")
    plt.ylabel("Accuratezza")
    plt.xticks(rotation=45, ha="right")

    # Aggiungi le etichette sopra le barre
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def save_results(accuracies: Dict[str, float], output_file: str = "results.txt"):
    """
    Salva i risultati su file
    """
    with open(output_file, "w") as f:
        f.write("Risultati della valutazione dei modelli\n")
        f.write("=====================================\n\n")
        for model_name, acc in accuracies.items():
            f.write(f"{model_name}: {acc:.4f}\n")


def main():
    try:
        # Directory contenente i modelli (modifica con il percorso corretto)
        model_dir = "models"

        # Carica tutti i modelli e il feature selector
        models, feature_selector = load_all_models(model_dir)
        if not models:
            print("Nessun modello trovato!")
            return

        # Carica i dati di test
        X_test, y_test = load_mnist_test_data(n_samples=1000)

        # Applica feature selection se disponibile
        if feature_selector is not None:
            X_test_selected = feature_selector.transform(X_test)
            print(
                f"Feature selection applicata: da {X_test.shape[1]} a {X_test_selected.shape[1]} features"
            )
        else:
            X_test_selected = X_test

        # Valuta tutti i modelli
        accuracies = {}
        predictions = {}

        for model_name, model in models.items():
            try:
                # Usa i dati appropriati per ogni modello
                if any(name in model_name.lower() for name in ["svm", "naive_bayes"]):
                    # Modelli che usano dati originali
                    X_test_curr = X_test
                    print(
                        f"Usando dati originali ({X_test.shape[1]} features) per {model_name}"
                    )
                else:
                    # Modelli che usano feature selection
                    X_test_curr = X_test_selected
                    print(
                        f"Usando dati con feature selection ({X_test_selected.shape[1]} features) per {model_name}"
                    )

                acc, y_pred = evaluate_model(
                    model, X_test_curr, y_test, model_name, model_dir
                )
                accuracies[model_name] = acc
                predictions[model_name] = y_pred
            except Exception as e:
                print(f"Errore nella valutazione del modello {model_name}: {str(e)}")
                continue

        if accuracies:
            # Visualizza il confronto delle accuratezze
            plot_accuracy_comparison(accuracies)

            # Visualizza alcune predizioni
            visualize_predictions(X_test, y_test, predictions)

            # Salva i risultati
            save_results(accuracies)
        else:
            print("Nessun modello è stato valutato con successo.")

    except Exception as e:
        print(f"Errore generale nell'esecuzione: {str(e)}")


if __name__ == "__main__":
    main()
