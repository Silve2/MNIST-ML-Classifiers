import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.special import betaln
import time
import os
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
import pickle
import seaborn as sns
import json

# Crea directory per i risultati
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_beta_naive_bayes_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Directory dei risultati creata in: {results_dir}")


class BetaNaiveBayes:
    def __init__(self, smooth_factor=1.0, feature_threshold=0.1):
        self.classes_ = None
        self.alpha_ = None
        self.beta_ = None
        self.class_priors_ = None
        self.smooth_factor = smooth_factor
        self.feature_threshold = feature_threshold

    def _preprocess_data(self, X):
        """
        Preprocessamento migliorato dei dati
        """
        # Applica threshold per ridurre il rumore
        X_thresh = np.where(X > self.feature_threshold, X, 0)

        # Normalizza per colonna per ridurre la variabilità tra feature
        X_norm = X_thresh / (np.max(X_thresh, axis=0) + 1e-6)

        # Clip per stabilità numerica
        eps = 1e-6
        return np.clip(X_norm, eps, 1 - eps)

    def _estimate_beta_params(self, X, prior_weight=0.1):
        """
        Stima migliorata dei parametri beta
        """
        n_samples = X.shape[0]

        # Calcola statistiche con smoothing
        means = (np.sum(X, axis=0) + prior_weight) / (n_samples + 2 * prior_weight)
        squared_means = (np.sum(X**2, axis=0) + prior_weight) / (
            n_samples + 2 * prior_weight
        )
        vars = squared_means - means**2

        # Assicura varianza minima
        vars = np.maximum(vars, 1e-4)

        # Calcola K con bounds più stretti
        K = (means * (1 - means) / vars) - 1
        K = np.clip(K, 0.5, 100)  # Bounds più conservativi

        # Calcola alpha e beta con smoothing addizionale
        alpha = K * means + self.smooth_factor
        beta = K * (1 - means) + self.smooth_factor

        return alpha, beta

    def fit(self, X, y):
        """
        Addestra il classificatore con bilanciamento delle classi
        """
        X = self._preprocess_data(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Inizializza parametri
        self.alpha_ = np.zeros((n_classes, n_features))
        self.beta_ = np.zeros((n_classes, n_features))

        # Calcola prior bilanciati
        class_counts = np.bincount(y, minlength=n_classes)
        max_count = np.max(class_counts)
        class_weights = max_count / (class_counts + 1e-6)
        self.class_priors_ = class_weights / np.sum(class_weights)

        # Per ogni classe, stima parametri con peso della classe
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            weight = class_weights[i]
            self.alpha_[i], self.beta_[i] = self._estimate_beta_params(
                X_c, prior_weight=weight * 0.1
            )

    def _beta_log_pdf(self, x, alpha, beta):
        """
        Log-pdf con stabilità numerica migliorata
        """
        x = np.clip(x, 1e-6, 1 - 1e-6)

        # Usa log1p per maggiore stabilità
        return (alpha - 1) * np.log(x) + (beta - 1) * np.log1p(-x) - betaln(alpha, beta)

    def predict_proba(self, X):
        """
        Calcola probabilità con normalizzazione migliorata
        """
        X = self._preprocess_data(X)

        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes_)))

        # Calcola log probabilities
        for i in range(len(self.classes_)):
            log_probs[:, i] = np.log(self.class_priors_[i])

            # Usa solo le feature più discriminative per questa classe
            feature_ll = self._beta_log_pdf(X, self.alpha_[i], self.beta_[i])

            # Normalizza i contributi delle feature
            log_probs[:, i] += np.sum(feature_ll, axis=1) / X.shape[1]

        # Normalizzazione stabile
        log_probs -= np.max(log_probs, axis=1)[:, np.newaxis]
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1)[:, np.newaxis]

        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def visualize_means(self, output_file=None):
        """
        Visualizza le medie dei parametri beta per ogni classe
        """
        n_classes = len(self.classes_)
        means = self.alpha_ / (self.alpha_ + self.beta_)

        # Crea una griglia di subplot più grande per una migliore visualizzazione
        rows = int(np.ceil(n_classes / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.ravel()

        for i in range(n_classes):
            im = axes[i].imshow(means[i].reshape(28, 28), cmap="viridis")
            axes[i].set_title(f"Classe {self.classes_[i]}")
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        # Disabilita gli assi rimanenti se ce ne sono
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_parameter_distributions(self, output_file=None):
        """
        Visualizza le distribuzioni dei parametri alpha e beta
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot delle distribuzioni di alpha
        sns.boxplot(data=self.alpha_, ax=ax1)
        ax1.set_title("Distribuzione dei parametri Alpha")
        ax1.set_ylabel("Valore")
        ax1.set_xlabel("Classe")

        # Plot delle distribuzioni di beta
        sns.boxplot(data=self.beta_, ax=ax2)
        ax2.set_title("Distribuzione dei parametri Beta")
        ax2.set_ylabel("Valore")
        ax2.set_xlabel("Classe")

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def save_model_and_visualizations(model, results_dir):
    """
    Salva il modello e crea tutte le visualizzazioni
    """
    # Crea la directory se non esiste
    os.makedirs(results_dir, exist_ok=True)

    # Salva il modello
    model_path = os.path.join(results_dir, "beta_naive_bayes_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modello salvato in: {model_path}")

    # Crea e salva le visualizzazioni
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Visualizza e salva le medie
    means_path = os.path.join(viz_dir, "class_means.png")
    model.visualize_means(means_path)
    print(f"Visualizzazione delle medie salvata in: {means_path}")

    # Visualizza e salva le distribuzioni dei parametri
    params_path = os.path.join(viz_dir, "parameter_distributions.png")
    model.visualize_parameter_distributions(params_path)
    print(f"Visualizzazione delle distribuzioni salvata in: {params_path}")


def plot_grid_search_results(grid_search_results, output_file):
    """
    Visualizza i risultati della grid search
    """
    smooth_factors = sorted(list(set(r["smooth_factor"] for r in grid_search_results)))
    thresholds = sorted(list(set(r["threshold"] for r in grid_search_results)))

    scores = np.zeros((len(smooth_factors), len(thresholds)))
    for r in grid_search_results:
        i = smooth_factors.index(r["smooth_factor"])
        j = thresholds.index(r["threshold"])
        scores[i, j] = r["score"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        scores,
        annot=True,
        fmt=".4f",
        xticklabels=thresholds,
        yticklabels=smooth_factors,
        cmap="viridis",
    )
    plt.xlabel("Feature Threshold")
    plt.ylabel("Smoothing Factor")
    plt.title("Grid Search Results")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_results(cv_results, output_file):
    """
    Visualizza i risultati della cross-validation
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy per fold
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(cv_results["scores"]) + 1), cv_results["scores"], "bo-")
    plt.axhline(
        y=cv_results["mean_score"],
        color="r",
        linestyle="--",
        label=f'Mean: {cv_results["mean_score"]:.4f}',
    )
    plt.fill_between(
        range(1, len(cv_results["scores"]) + 1),
        np.array(cv_results["scores"]) - cv_results["std_score"],
        np.array(cv_results["scores"]) + cv_results["std_score"],
        alpha=0.2,
    )
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy per Fold")
    plt.legend()

    # Plot training times
    plt.subplot(1, 2, 2)
    train_times = [t["train_time"] for t in cv_results["times"]]
    pred_times = [t["pred_time"] for t in cv_results["times"]]
    plt.plot(range(1, len(train_times) + 1), train_times, "bo-", label="Training Time")
    plt.plot(range(1, len(pred_times) + 1), pred_times, "ro-", label="Prediction Time")
    plt.xlabel("Fold")
    plt.ylabel("Time (seconds)")
    plt.title("Training and Prediction Times per Fold")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model():
    print("Caricamento MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = np.asarray(X.astype("float32")) / 255.0
    y = np.asarray(y.astype("int32"))

    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Grid search sui parametri
    best_score = 0
    best_params = None

    smooth_factors = [0.5, 1.0, 2.0]
    thresholds = [0.05, 0.1, 0.2]

    # Usa un subset per grid search
    subset_size = 60000
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_subset = X_train[indices]
    y_subset = y_train[indices]

    # Salva i risultati della grid search
    grid_search_results = []

    print("\nGrid Search sui parametri...")
    for smooth in smooth_factors:
        for thresh in thresholds:
            model = BetaNaiveBayes(smooth_factor=smooth, feature_threshold=thresh)
            model.fit(X_subset, y_subset)
            score = accuracy_score(y_subset, model.predict(X_subset))

            result = {"smooth_factor": smooth, "threshold": thresh, "score": score}
            grid_search_results.append(result)

            if score > best_score:
                best_score = score
                best_params = {"smooth_factor": smooth, "feature_threshold": thresh}

            print(f"Smooth: {smooth}, Threshold: {thresh}, Score: {score:.4f}")

    # Salva risultati grid search
    with open(os.path.join(results_dir, "grid_search_results.json"), "w") as f:
        json.dump(grid_search_results, f, indent=4)

    # Plot grid search results
    plot_grid_search_results(
        grid_search_results, os.path.join(results_dir, "grid_search_results.png")
    )

    print(f"\nMigliori parametri: {best_params}")

    # 10-fold CV con i migliori parametri
    print("\nEsecuzione 10-fold CV...")
    cv_scores = []
    cv_times = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train[train_idx]


def main():
    # Crea directory per i risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_beta_naive_bayes_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Directory dei risultati creata in: {results_dir}")

    try:
        print("Caricamento MNIST...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        print("Dataset MNIST caricato con successo!")

        if X is None or y is None:
            raise ValueError("Errore nel caricamento del dataset MNIST")

        X = np.asarray(X.astype("float32")) / 255.0
        y = np.asarray(y.astype("int32"))

        print(f"Dimensione del dataset: {X.shape}")
        print(f"Numero di classi uniche: {len(np.unique(y))}")

        # Divide in train e test
        train_size = 60000
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Grid search parameters
        smooth_factors = [0.5, 1.0, 2.0]
        thresholds = [0.05, 0.1, 0.2]

        # Usa un subset per grid search
        subset_size = 60000
        print(f"\nCreazione subset per grid search (size={subset_size})...")
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]

        # Grid search
        print("\nInizio Grid Search...")
        grid_search_results = []
        best_score = 0
        best_params = None

        for smooth in smooth_factors:
            for thresh in thresholds:
                print(f"\nTestando smooth={smooth}, threshold={thresh}")
                try:
                    model = BetaNaiveBayes(
                        smooth_factor=smooth, feature_threshold=thresh
                    )
                    model.fit(X_subset, y_subset)
                    score = accuracy_score(y_subset, model.predict(X_subset))

                    result = {
                        "smooth_factor": smooth,
                        "threshold": thresh,
                        "score": float(score),  
                    }
                    grid_search_results.append(result)

                    print(f"Score ottenuto: {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "smooth_factor": smooth,
                            "feature_threshold": thresh,
                        }

                except Exception as e:
                    print(
                        f"Errore durante grid search per smooth={smooth}, thresh={thresh}: {str(e)}"
                    )
                    continue

        # Salva risultati grid search
        try:
            grid_search_path = os.path.join(results_dir, "grid_search_results.json")
            with open(grid_search_path, "w") as f:
                json.dump(grid_search_results, f, indent=4)
            print(f"\nRisultati grid search salvati in: {grid_search_path}")

            # Training finale
            if best_params:
                print(f"\nMigliori parametri trovati: {best_params}")
                print("\nTraining modello finale...")
                final_model = BetaNaiveBayes(**best_params)
                final_model.fit(X_train, y_train)

                # Salva il modello e le visualizzazioni
                save_model_and_visualizations(final_model, results_dir)

                # Test finale
                y_pred = final_model.predict(X_test)
                final_score = accuracy_score(y_test, y_pred)
                print(f"\nAccuracy finale sul test set: {final_score:.4f}")

            else:
                print("\nERRORE: Grid search non ha trovato parametri validi")

        except Exception as e:
            print(f"\nErrore durante il salvataggio dei risultati: {str(e)}")

    except Exception as e:
        print(f"\nErrore generale durante l'esecuzione: {str(e)}")


if __name__ == "__main__":
    # Esegui il main con gestione delle eccezioni
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruzione manuale del programma")
    except Exception as e:
        print(f"\nErrore critico: {str(e)}")
    finally:
        print("\nProgramma terminato")
