"""Funzioni di plotting condivise da `run_pipeline.py` e dai notebook
(si vedano le sezioni "Configurazione sperimentale", "Risultati:
Wine" e "Analisi comparativa trasversale" del Capitolo 7)"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, title: str, out_path: Path,
                           cmap: str = "Blues") -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(np.asarray(confusion_matrix), annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Classe predetta")
    ax.set_ylabel("Classe reale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices_side_by_side(qsvc_cm, vqc_cm, dataset_name: str,
                                          out_path: Path) -> None:
    """Matrici di confusione affiancate per QSVC e VQC (Figura
    `confusion_breast_cancer_ideale`, sezione "Risultati: Breast Cancer
    Wisconsin", sottosezione "Simulazione ideale")."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, cm, title, cmap in zip(
        axes, (qsvc_cm, vqc_cm), ("QSVC", "VQC"), ("Blues", "Purples")
    ):
        sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(f"{title} - {dataset_name} (simulazione ideale)")
        ax.set_xlabel("Classe predetta")
        ax.set_ylabel("Classe reale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vqc_convergence(cost_histories: dict, out_path: Path) -> None:
    """Curva di convergenza della funzione di costo del VQC per dataset
    (Figura `convergenza_vqc`, sezione "Configurazione sperimentale").

    Un subplot per dataset, ciascuno con la propria scala dell'asse y:
    i valori assoluti della funzione di costo non sono confrontabili tra
    dataset (dipendono dal numero di qubit e dalla dimensione del
    training set), per cui un unico asse condiviso schiaccerebbe la
    discesa dei dataset con costo più basso contro quello, molto più
    alto, di Digits."""
    names = [name for name, history in cost_histories.items() if history]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        history = cost_histories[name]
        ax.plot(range(1, len(history) + 1), history, color="tab:blue")
        ax.set_xlabel("Iterazione")
        ax.set_ylabel("Funzione di costo")
        ax.set_title(name)
    fig.suptitle("Curva di convergenza del VQC (COBYLA, simulazione ideale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_accuracy_comparison(rows: list[dict], dataset_name: str, out_path: Path,
                                      model_order: list[str] = None) -> None:
    """Grafico a barre dell'accuratezza dei modelli su un singolo dataset,
    con barre di errore per i modelli classici (deviazione standard
    dell'accuratezza tra i fold di cross-validation, quando disponibile
    tramite la chiave `accuracy_cv_std`; assente per QSVC/VQC, che non
    usano cross-validation) (Figura `wine_accuracy_comparison`,
    sezione "Risultati: Wine")."""
    order = model_order or ["Logistic Regression", "SVM (RBF kernel)",
                             "Random Forest", "QSVC", "VQC"]
    by_model = {r["model"]: r for r in rows}
    models = [m for m in order if m in by_model]
    accuracies = [by_model[m]["accuracy"] for m in models]
    errors = [by_model[m].get("accuracy_cv_std") or 0 for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(models, accuracies, yerr=errors, capsize=4,
           color=sns.color_palette("Blues_d", n_colors=len(models)))
    ax.set_ylabel("Accuratezza di test")
    ax.set_xlabel("Modello")
    ax.set_title(f"Confronto dell'accuratezza sul dataset {dataset_name} "
                 f"(simulazione ideale)")
    ax.set_ylim(0, 1.05)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_ENVIRONMENT_LABELS = {
    "ideal": "Ideale",
    "noisy_simulation": "Rumore",
    "real_hardware": "Hardware reale",
}
_ENVIRONMENT_ORDER = ["ideal", "noisy_simulation", "real_hardware"]


def plot_noise_hardware_degradation(rows: list[dict], out_path: Path) -> None:
    """Degrado dell'accuratezza di QSVC e VQC al crescere del realismo
    dell'ambiente di esecuzione (ideale, rumore, hardware reale), una
    linea per dataset (Figura `degrado_performance`, sezione "Effetto del
    rumore e del passaggio a hardware reale").

    `rows` è una lista di dizionari con chiavi `dataset`, `model`
    (`"QSVC"` o `"VQC"`), `environment` (una tra `ideal`,
    `noisy_simulation`, `real_hardware`) e `accuracy`. Un ambiente
    mancante per un dato dataset/modello viene semplicemente omesso
    dalla linea corrispondente, così da poter tracciare il grafico
    anche con una campagna sperimentale solo parzialmente completata.
    """
    models = [m for m in ("QSVC", "VQC") if any(r["model"] == m for r in rows)]
    datasets = sorted({r["dataset"] for r in rows})

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for dataset in datasets:
            by_env = {
                r["environment"]: r["accuracy"] for r in rows
                if r["model"] == model and r["dataset"] == dataset
            }
            envs = [e for e in _ENVIRONMENT_ORDER if e in by_env]
            if not envs:
                continue
            ax.plot(envs, [by_env[e] for e in envs], marker="o", label=dataset)
        ax.set_xticks(_ENVIRONMENT_ORDER)
        ax.set_xticklabels([_ENVIRONMENT_LABELS[e] for e in _ENVIRONMENT_ORDER])
        ax.set_ylabel("Accuratezza di test")
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Dataset")
    fig.suptitle("Degrado delle performance al crescere del realismo dell'ambiente di esecuzione")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(global_accuracy_table, out_path: Path,
                              title: str = "Confronto globale dell'accuratezza "
                                            "(simulazione ideale)") -> None:
    """Grafico a barre raggruppate dell'accuratezza per modello e dataset
    (Figura `confronto_globale_accuratezza`, sezione "Confronto
    dell'accuratezza tra dataset e modelli")."""
    fig, ax = plt.subplots(figsize=(8, 5))
    global_accuracy_table.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuratezza di test")
    ax.set_xlabel("Modello")
    ax.set_title(title)
    ax.legend(title="Dataset")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
