"""Script di orchestrazione della pipeline sperimentale completa.

Esegue, per uno o piu' dataset tra `breast_cancer`, `wine` e
`digits`, l'intera sequenza descritta nel Capitolo 7 (acquisizione,
preprocessing, addestramento dei modelli classici di baseline e dei
modelli quantistici QSVC/VQC in simulazione ideale, valutazione) e
salva i risultati in formato tabellare in `results/tables` e le
figure corrispondenti in `results/figures`.

Questo script è un orchestratore: la logica di addestramento e
valutazione dei modelli vive in `src/pipeline.py` (e nei moduli
`src.classical`/`src.quantum`/`src.execution`/`src.evaluation`
da cui dipende), in modo da essere riutilizzata dai notebook di
sperimentazione (si veda `notebooks/03_classical_baseline.ipynb` e
seguenti) senza duplicazione di codice.

Uso:
    - python run_pipeline.py --datasets wine
    - python run_pipeline.py --datasets breast_cancer
    - python run_pipeline.py --datasets digits
    - python run_pipeline.py --aggregate
    - python run_pipeline.py --digits-pca-sensitivity
    - python run_pipeline.py --optimizer-comparison

Questo script si limita alla modalità di simulazione ideale
(`StatevectorSampler`, si veda `src/execution/backend_manager.py`);
per le modalità di simulazione con rumore ed esecuzione su hardware
reale si veda il notebook `06_hardware_execution.ipynb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    DATASET_CONFIGS,
    DIGITS_PCA_SENSITIVITY_COMPONENTS,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    VQC_MAXITER,
    QSVC_IDEAL_MAX_CIRCUITS_PER_JOB,
)
from src.pipeline import (
    load_and_preprocess_dataset,
    train_classical_models,
    train_and_evaluate_qsvc,
    train_and_evaluate_vqc,
    aggregate_ideal_results,
    update_degradation_figure,
)
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_confusion_matrices_side_by_side,
    plot_vqc_convergence,
    plot_dataset_accuracy_comparison,
)

RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_ORDER = ["Logistic Regression", "SVM (RBF kernel)", "Random Forest", "QSVC", "VQC"]


def _load_and_preprocess(name: str, n_components: int):
    return load_and_preprocess_dataset(
        name, n_components, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def run_dataset(name: str) -> None:
    cfg = DATASET_CONFIGS[name]
    print(f"[{name}] Acquisizione, validazione e preprocessing "
          f"(Min-Max + PCA a {cfg['n_components']} componenti)...")
    X_train, X_test, y_train, y_test, _pca = _load_and_preprocess(name, cfg["n_components"])

    print(f"[{name}] Addestramento modelli classici (grid search {CV_FOLDS}-fold)...")
    rows = train_classical_models(X_train, y_train, X_test, y_test,
                                   random_state=RANDOM_STATE, cv_folds=CV_FOLDS)

    print(f"[{name}] Addestramento QSVC (simulazione ideale)...")
    rows.append(train_and_evaluate_qsvc(
        cfg["n_components"], cfg["feature_map_reps"], cfg["entanglement"],
        X_train, y_train, X_test, y_test, random_state=RANDOM_STATE,
        max_circuits_per_job=QSVC_IDEAL_MAX_CIRCUITS_PER_JOB,
    ))

    print(f"[{name}] Addestramento VQC (simulazione ideale, COBYLA, "
          f"maxiter={VQC_MAXITER})...")
    vqc_row, cost_history = train_and_evaluate_vqc(
        cfg["n_components"], cfg["feature_map_reps"], cfg["ansatz_reps"],
        cfg["entanglement"], X_train, y_train, X_test, y_test, maxiter=VQC_MAXITER,
        random_state=RANDOM_STATE,
    )
    rows.append(vqc_row)

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / f"results_ideal_{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{name}] Risultati salvati in {out_path}")

    if cost_history:
        cost_path = TABLES_DIR / f"vqc_cost_history_{name}.csv"
        pd.DataFrame({"iteration": range(1, len(cost_history) + 1),
                      "cost": cost_history}).to_csv(cost_path, index=False)
        print(f"[{name}] Curva di convergenza VQC salvata in {cost_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    qsvc_cm = next(r["confusion_matrix"] for r in rows if r["model"] == "QSVC")
    vqc_cm = next(r["confusion_matrix"] for r in rows if r["model"] == "VQC")
    plot_confusion_matrix(qsvc_cm, f"QSVC - {name} (simulazione ideale)",
                           FIGURES_DIR / f"confusion_matrix_qsvc_{name}_ideale.png",
                           cmap="Blues")
    plot_confusion_matrix(vqc_cm, f"VQC - {name} (simulazione ideale)",
                           FIGURES_DIR / f"confusion_matrix_vqc_{name}_ideale.png",
                           cmap="Purples")
    if name == "breast_cancer":
        plot_confusion_matrices_side_by_side(
            qsvc_cm, vqc_cm, name,
            FIGURES_DIR / "confusion_matrix_breast_cancer_ideale.png",
        )
    if name == "wine":
        plot_dataset_accuracy_comparison(
            rows, "Wine", FIGURES_DIR / "wine_accuracy_comparison.png",
            model_order=MODEL_ORDER,
        )
    print(f"[{name}] Matrici di confusione salvate in {FIGURES_DIR}")


def run_digits_pca_sensitivity() -> None:
    """Riproduce l'analisi di sensitività del QSVC sul dataset Digits al
    variare del numero di componenti PCA (Tabella
    `digits_sensitivity_pca`, sezione "Effetto della riduzione
    dimensionale")."""
    name = "digits"
    cfg = DATASET_CONFIGS[name]

    rows = []
    for n_components in DIGITS_PCA_SENSITIVITY_COMPONENTS:
        X_train, X_test, y_train, y_test, pca = _load_and_preprocess(name, n_components)
        result = train_and_evaluate_qsvc(
            n_components, cfg["feature_map_reps"], cfg["entanglement"],
            X_train, y_train, X_test, y_test, random_state=RANDOM_STATE,
            max_circuits_per_job=QSVC_IDEAL_MAX_CIRCUITS_PER_JOB,
        )
        rows.append({
            "n_components": n_components,
            "n_qubits": n_components,
            "explained_variance": pca.explained_variance_ratio_.sum(),
            "qsvc_accuracy": result["accuracy"],
        })

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "digits_pca_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"Analisi di sensitività PCA (Digits) salvata in {out_path}")


def run_optimizer_comparison() -> None:
    """Confronta COBYLA e SPSA per l'addestramento del VQC sul dataset
    Breast Cancer Wisconsin in simulazione ideale (Tabella
    `confronto_ottimizzatori`, sezione "Confronto tra ottimizzatori per il VQC").

    Il tempo riportato (`total_time_s`) è il solo tempo di fit, misurato
    con `timed_fit_predict` come per gli altri esperimenti (Sezione
    7.6.2): è la fase su cui incide la scelta dell'ottimizzatore, a
    differenza del tempo di predict che è indipendente da essa.

    Per un confronto equo dell'inizializzazione, `algorithm_globals.random_seed`
    è fissato a `RANDOM_STATE` immediatamente prima di ciascuna delle due
    costruzioni del VQC, in modo che COBYLA e SPSA partano dallo stesso
    punto iniziale casuale dei parametri variazionali. Questo non basta
    però a rendere equo il *budget* di valutazione tra i due ottimizzatori:
    in qiskit-machine-learning, `COBYLA.maxiter` conta il numero massimo di
    valutazioni della funzione di costo, mentre `SPSA.maxiter` conta il
    numero di iterazioni, ciascuna delle quali richiede almeno due
    valutazioni (oltre a un'eventuale fase di calibrazione iniziale), le
    due colonne `n_iterations` non sono quindi direttamente confrontabili
    a parità di `maxiter=VQC_MAXITER`. Per questo il sampler è avvolto in
    un `CountingSampler` (come per `train_and_evaluate_vqc`) anche in
    questo confronto: `n_circuits`, il numero di circuiti quantistici
    effettivamente eseguiti, è la base di confronto del budget realmente
    omogenea tra i due ottimizzatori."""
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import SPSA
    from qiskit_machine_learning.utils import algorithm_globals

    from src.quantum.feature_maps import build_feature_map
    from src.quantum.ansatz import build_ansatz
    from src.quantum.vqc_model import build_vqc
    from src.execution.backend_manager import get_sampler, CountingSampler
    from src.evaluation.metrics import timed_fit_predict, evaluate_model

    name = "breast_cancer"
    cfg = DATASET_CONFIGS[name]
    X_train, X_test, y_train, y_test, _pca = _load_and_preprocess(name, cfg["n_components"])

    feature_map = build_feature_map(cfg["n_components"], reps=cfg["feature_map_reps"],
                                     entanglement=cfg["entanglement"])
    ansatz = build_ansatz(cfg["n_components"], reps=cfg["ansatz_reps"],
                           entanglement=cfg["entanglement"])

    rows = []
    for opt_name, optimizer in (
        ("COBYLA", None),  # costruito internamente da build_vqc
        ("SPSA", SPSA(maxiter=VQC_MAXITER)),
    ):
        sampler = CountingSampler(get_sampler("ideal", seed=RANDOM_STATE))
        cost_history: list[float] = []

        if optimizer is None:
            # COBYLA non ha un proprio attributo `callback`: VQC lo invoca
            # con la convenzione generica a due argomenti (pesi, valore).
            def callback(_weights, cost):
                cost_history.append(float(cost))

            vqc = build_vqc(feature_map, ansatz, sampler, maxiter=VQC_MAXITER,
                             callback=callback, random_state=RANDOM_STATE)
        else:
            # SPSA ha un proprio attributo `callback` nativo: VQC lo
            # rileva (`hasattr(optimizer, "callback")`) e vi assegna
            # direttamente la funzione fornita, che viene quindi
            # invocata con la firma nativa di SPSA a cinque argomenti
            # posizionali (non quella generica a due usata da COBYLA).
            def callback(_nfev, _params, cost, _update_norm, _accepted):
                cost_history.append(float(cost))

            algorithm_globals.random_seed = RANDOM_STATE
            vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz,
                      optimizer=optimizer, callback=callback)

        y_pred, fit_time, _predict_time = timed_fit_predict(vqc, X_train, y_train, X_test)
        test_accuracy = evaluate_model(y_test, y_pred)["accuracy"]

        rows.append({
            "optimizer": opt_name,
            "test_accuracy": test_accuracy,
            "n_iterations": len(cost_history),
            "n_circuits": sampler.n_circuits,
            "total_time_s": fit_time,
        })

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "optimizer_comparison_breast_cancer.csv"
    df.to_csv(out_path, index=False)
    print(f"Confronto ottimizzatori (Breast Cancer) salvato in {out_path}")


def run_aggregate() -> None:
    """Aggrega le tabelle disponibili in `results/tables` (indipendentemente
    da quali dataset siano già stati eseguiti) nelle tabelle e figure
    comparative del Capitolo 7, sezione "Analisi comparativa
    trasversale": confronto globale
    dell'accuratezza e curva di convergenza del VQC. I dataset non ancora
    eseguiti vengono semplicemente omessi, così da poter rilanciare questa
    funzione in modo incrementale dopo ogni `run_dataset`.

    La logica di aggregazione vera e propria (tabella/figura
    `confronto_globale_accuratezza`, Figura `degrado_performance`) vive in
    `aggregate_ideal_results`/`update_degradation_figure` (`src/pipeline.py`),
    riutilizzate identicamente dal notebook `07_results_comparison.ipynb`."""
    result_files = sorted(TABLES_DIR.glob("results_ideal_*.csv"))
    if not result_files:
        print("Nessun risultato disponibile in results/tables: eseguire prima "
              "almeno un dataset con --datasets.")
        return

    combined_cols = ["dataset", "model", "accuracy", "precision", "recall",
                      "f1_score", "fit_time_s"]
    frames: list[pd.DataFrame] = []
    for f in result_files:
        name = f.stem.replace("results_ideal_", "")
        df = pd.read_csv(f)
        df["dataset"] = name
        frames.append(df[[c for c in combined_cols if c in df.columns]])
    combined = pd.concat(frames, ignore_index=True)

    aggregate_ideal_results(combined, TABLES_DIR, FIGURES_DIR, model_order=MODEL_ORDER)
    print(f"Confronto globale dell'accuratezza aggiornato "
          f"({', '.join(f.stem.replace('results_ideal_', '') for f in result_files)}).")

    cost_files = sorted(TABLES_DIR.glob("vqc_cost_history_*.csv"))
    if cost_files:
        cost_histories = {
            f.stem.replace("vqc_cost_history_", ""): pd.read_csv(f)["cost"].tolist()
            for f in cost_files
        }
        plot_vqc_convergence(cost_histories, FIGURES_DIR / "vqc_convergenza.png")
        print(f"Curva di convergenza VQC aggiornata "
              f"({', '.join(cost_histories.keys())}).")

    if update_degradation_figure(combined, TABLES_DIR, FIGURES_DIR):
        print("Figura di degrado delle performance (ideale/rumore/hardware) aggiornata.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()),
        default=[],
        help="Sottoinsieme di dataset su cui eseguire la pipeline "
             "(eseguibili anche uno alla volta, in run separate).",
    )
    parser.add_argument(
        "--digits-pca-sensitivity", action="store_true",
        help="Esegue anche l'analisi di sensitività PCA sul dataset Digits.",
    )
    parser.add_argument(
        "--optimizer-comparison", action="store_true",
        help="Esegue anche il confronto COBYLA/SPSA sul dataset Breast Cancer.",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Rigenera le tabelle/figure comparative a partire dai risultati "
             "già presenti in results/tables (anche parziali).",
    )
    args = parser.parse_args()

    if not (args.datasets or args.digits_pca_sensitivity
            or args.optimizer_comparison or args.aggregate):
        print("Nessuna azione richiesta: specificare --datasets, "
              "--digits-pca-sensitivity, --optimizer-comparison e/o --aggregate.")
        return

    for name in args.datasets:
        run_dataset(name)

    if args.digits_pca_sensitivity:
        run_digits_pca_sensitivity()

    if args.optimizer_comparison:
        run_optimizer_comparison()

    if args.aggregate:
        run_aggregate()


if __name__ == "__main__":
    main()
