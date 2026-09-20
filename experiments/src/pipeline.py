"""Funzioni di alto livello per una singola run sperimentale (un modello,
un dataset): costruiscono il modello a partire dai moduli di
`src.classical`/`src.quantum`, lo addestrano, lo valutano e ne
misurano il tempo di esecuzione. Include inoltre le funzioni di
aggregazione dei risultati di più dataset/modelli nelle tabelle e
figure comparative del Capitolo 4 (sezione "Analisi comparativa
trasversale"): `aggregate_ideal_results` e `update_degradation_figure`.

Vengono riutilizzate sia da `run_pipeline.py` (orchestratore
da riga di comando, sezione "Struttura del progetto e ambiente
containerizzato") sia dai notebook di sperimentazione
(`02_preprocessing.ipynb`, `03_classical_baseline.ipynb`,
`04_quantum_qsvc.ipynb`, `05_quantum_vqc.ipynb`,
`06_hardware_execution.ipynb`, `07_results_comparison.ipynb`), in modo
che le due modalità di esecuzione della campagna sperimentale (script
da riga di comando o notebook interattivi) condividano un'unica
implementazione della logica di aggregazione, invece di duplicarla.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.acquisition import load_dataset
from src.data.preprocessing import preprocess
from src.classical.baseline_models import get_classical_models, GRID_SEARCH_PARAMS
from src.quantum.feature_maps import build_feature_map
from src.quantum.ansatz import build_ansatz
from src.quantum.qsvc_model import build_qsvc
from src.quantum.vqc_model import build_vqc
from src.execution.backend_manager import get_sampler, CountingSampler
from src.evaluation.metrics import evaluate_model, timed_fit_predict, circuit_stats
from src.evaluation.plots import plot_accuracy_comparison, plot_noise_hardware_degradation


def load_and_preprocess_dataset(name: str, n_components: int,
                                 test_size: float = 0.2, random_state: int = 42):
    """Acquisisce un dataset e lo preprocessa (Min-Max + PCA), pronto per
    l'addestramento. Usata da `run_pipeline.py`, dal notebook
    `02_preprocessing.ipynb` (che la usa per generare gli artefatti
    intermedi `data/processed/*.npz`) e dal notebook
    `06_hardware_execution.ipynb`, che invece deve poter eseguire una
    singola configurazione dataset/modello senza dipendere da quegli
    stessi artefatti.

    Le etichette sono convertite in `numpy.ndarray`: il VQC richiede
    questo tipo (l'one-hot encoding interno di qiskit-machine-learning
    chiama `.reshape`, non supportato da una `pandas.Series`)."""
    X, y = load_dataset(name)
    X_train, X_test, y_train, y_test, _scaler, pca = preprocess(
        X, y, n_components=n_components, test_size=test_size, random_state=random_state,
    )
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), pca


def train_classical_models(X_train, y_train, X_test, y_test,
                            random_state: int = 42, cv_folds: int = 5) -> list[dict]:
    """Addestra i tre modelli classici di baseline con grid search e
    validazione incrociata stratificata (sezione "Modelli classici di
    baseline", Tabella `grid_search_classici`), restituendo le metriche di valutazione sul
    test set.

    Include anche `accuracy_cv_std`, la deviazione standard dell'accuratezza
    tra i fold di cross-validation per la combinazione di iperparametri
    scelta (`GridSearchCV.cv_results_["std_test_score"]`): usata come barra
    di errore nel grafico di confronto per singolo dataset (Figura
    `wine_accuracy_comparison`, sezione "Risultati: Wine"). Non è disponibile per QSVC/VQC,
    che non usano cross-validation.

    `fit_time_s`/`predict_time_s` misurano, come per QSVC e VQC (si veda
    `train_and_evaluate_qsvc`), il solo fit/predict del modello finale
    (agli iperparametri migliori trovati dalla grid search), non l'intera
    grid search: quest'ultima addestra fino a decine di modelli per fold
    (ad esempio 4x5x5=100 fit per SVM), un costo di natura diversa dal
    singolo fit di un modello già configurato, e non sarebbe altrimenti
    confrontabile con il tempo di fit dei modelli quantistici. Il costo
    della ricerca degli iperparametri è comunque tracciato separatamente
    in `grid_search_time_s` (sezione "Costo computazionale e tempi di
    esecuzione")."""
    rows = []
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for name, model in get_classical_models(random_state).items():
        search = GridSearchCV(
            model, GRID_SEARCH_PARAMS[name], cv=cv, scoring="accuracy", n_jobs=-1
        )
        start_search = time.perf_counter()
        search.fit(X_train, y_train)
        grid_search_time = time.perf_counter() - start_search
        accuracy_cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

        # Fit/predict del solo modello finale (best_params_), per una
        # misura omogenea con il fit singolo di QSVC/VQC; si parte da un
        # clone non addestrato di best_estimator_ anziché riutilizzarne lo
        # stato già fittato dalla grid search.
        best_model = clone(search.best_estimator_)
        y_pred, fit_time, predict_time = timed_fit_predict(
            best_model, X_train, y_train, X_test
        )

        metrics = evaluate_model(y_test, y_pred)
        rows.append({
            "model": name,
            "best_params": json.dumps(search.best_params_),
            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
            "grid_search_time_s": grid_search_time,
            "accuracy_cv_std": accuracy_cv_std,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        })
    return rows


def train_and_evaluate_qsvc(n_qubits, reps, entanglement, X_train, y_train, X_test, y_test,
                             sampler_mode: str = "ideal", noise_model=None,
                             backend_name: str = None, pass_manager=None,
                             random_state: int = 42, max_circuits_per_job: int = None) -> dict:
    """Costruisce, addestra e valuta un QSVC (sezione "Modello QSVC") sul backend
    indicato da `sampler_mode` (si veda `src.execution.backend_manager.get_sampler`: `ideal`,
    `noisy_simulation` o `real_hardware`).

    Il tempo di fit/predict è misurato con `timed_fit_predict`, la stessa
    funzione usata per i modelli classici (sezione "Modulo di
    valutazione"), in modo da
    garantire un confronto omogeneo dei tempi di esecuzione (Tabella
    `costo_computazionale`, sezione "Costo computazionale e tempi di
    esecuzione"). Il sampler è wrappato in un
    `CountingSampler` per contare il numero di circuiti quantistici
    effettivamente eseguiti durante fit/predict (colonna "N. circuiti
    eseguiti" della stessa tabella). L'accuratezza sul train è
    un'informazione diagnostica aggiuntiva (utile per individuare
    overfitting/underfitting), calcolata separatamente e non inclusa nel
    tempo misurato né nel conteggio dei circuiti.

    Per `sampler_mode` diverso da `"ideal"` è necessario passare anche
    `pass_manager` (si veda `src.execution.backend_manager.get_pass_manager`):
    a differenza di `StatevectorSampler`, `AerSampler` e l'esecuzione su
    hardware reale richiedono circuiti già transpilati nel basis gate set
    del backend.

    `random_state` fissa il seed del campionamento a shot finiti del
    sampler ideale/con rumore (`StatevectorSampler`/`AerSampler`
    campionano di default 1024 shot da un generatore non seedato per
    default), necessario per la riproducibilità della stima del kernel
    quantistico; ignorato in modalità `real_hardware`.

    `max_circuits_per_job`, inoltrato a `build_qsvc`, suddivide il
    calcolo della matrice di kernel in più chiamate più piccole invece
    di una singola chiamata con una coppia di circuiti per ciascuna
    coppia di campioni di addestramento: necessario per `sampler_mode`
    diverso da `"ideal"`, dove una chiamata unica con training set di
    alcune centinaia di campioni può esaurire la memoria disponibile
    (si veda la sezione "Limiti dello studio e minacce alla validità").
    """
    feature_map = build_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    sampler = CountingSampler(
        get_sampler(sampler_mode, noise_model=noise_model, backend_name=backend_name,
                    seed=random_state)
    )
    qsvc = build_qsvc(feature_map, sampler, pass_manager=pass_manager,
                       max_circuits_per_job=max_circuits_per_job)

    y_pred, fit_time, predict_time = timed_fit_predict(qsvc, X_train, y_train, X_test)
    n_circuits = sampler.n_circuits
    train_accuracy = qsvc.score(X_train, y_train)

    metrics = evaluate_model(y_test, y_pred)
    stats = circuit_stats(feature_map)
    return {
        "model": "QSVC",
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "n_circuits": n_circuits,
        "train_accuracy": train_accuracy,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "circuit_depth": stats["depth"],
        "n_qubits": n_qubits,
    }


def train_and_evaluate_vqc(n_qubits, fm_reps, ansatz_reps, entanglement,
                            X_train, y_train, X_test, y_test, maxiter: int = 100,
                            sampler_mode: str = "ideal", noise_model=None,
                            backend_name: str = None, pass_manager=None,
                            random_state: int = 42) -> tuple[dict, list]:
    """Costruisce, addestra e valuta un VQC (sezione "Modello VQC") sul backend
    indicato da `sampler_mode`, restituendo anche la cronologia della
    funzione di costo raccolta tramite callback (Figura
    `convergenza_vqc`).

    Come per `train_and_evaluate_qsvc`, il tempo di fit/predict è misurato
    con `timed_fit_predict` (stessa funzione dei modelli classici), per un
    confronto omogeneo dei tempi di esecuzione (sezione "Costo
    computazionale e tempi di esecuzione"), il
    sampler è avvolto in un `CountingSampler` per contare i circuiti
    eseguiti durante fit/predict, e per `sampler_mode` diverso da
    `"ideal"` è necessario passare anche `pass_manager` (si veda
    `train_and_evaluate_qsvc`).

    `random_state` fissa sia il seed del campionamento a shot finiti del
    sampler (si veda `train_and_evaluate_qsvc`) sia, tramite `build_vqc`,
    il seed di `algorithm_globals.random` usato per generare il punto
    iniziale casuale dei parametri variazionali: senza di esso
    l'addestramento del VQC non è riproducibile tra esecuzioni diverse.
    """
    feature_map = build_feature_map(n_qubits, reps=fm_reps, entanglement=entanglement)
    ansatz = build_ansatz(n_qubits, reps=ansatz_reps, entanglement=entanglement)
    sampler = CountingSampler(
        get_sampler(sampler_mode, noise_model=noise_model, backend_name=backend_name,
                    seed=random_state)
    )

    cost_history: list[float] = []

    def callback(_weights, cost):
        cost_history.append(float(cost))

    vqc = build_vqc(feature_map, ansatz, sampler, maxiter=maxiter, callback=callback,
                     pass_manager=pass_manager, random_state=random_state)

    y_pred, fit_time, predict_time = timed_fit_predict(vqc, X_train, y_train, X_test)
    n_circuits = sampler.n_circuits
    train_accuracy = vqc.score(X_train, y_train)

    metrics = evaluate_model(y_test, y_pred)
    full_circuit = feature_map.compose(ansatz)
    stats = circuit_stats(full_circuit)
    row = {
        "model": "VQC",
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "n_circuits": n_circuits,
        "train_accuracy": train_accuracy,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "circuit_depth": stats["depth"],
        "n_qubits": n_qubits,
    }
    return row, cost_history


def aggregate_ideal_results(combined: pd.DataFrame, tables_dir: Path, figures_dir: Path,
                             model_order: list[str] | None = None) -> pd.DataFrame:
    """Aggrega, in un'unica tabella e figura, i risultati in simulazione ideale di più
    dataset e modelli (Tabella/Figura `confronto_globale_accuratezza`, sezione "Confronto
    dell'accuratezza tra dataset e modelli", Capitolo 4).

    `combined` deve avere almeno le colonne `dataset`, `model` e `accuracy` (una riga per
    coppia dataset/modello); colonne aggiuntive (`precision`, `recall`, `f1_score`,
    `fit_time_s`, `accuracy_cv_std`, ...) vengono preservate in `combined_ideal_results.csv`
    ma non utilizzate da questa funzione. `model_order`, se fornito, fissa l'ordine delle
    righe della tabella pivot risultante (modello x dataset); i modelli assenti da
    `combined` vengono semplicemente omessi, così da poter aggregare anche una campagna
    sperimentale solo parzialmente completata.

    Riutilizzata identicamente da `run_pipeline.py --aggregate` e dal notebook
    `07_results_comparison.ipynb`, evitando di duplicare la logica di aggregazione tra i
    due percorsi (si veda "Struttura del progetto e ambiente containerizzato").
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tables_dir / "combined_ideal_results.csv", index=False)

    global_accuracy = combined.pivot(index="model", columns="dataset", values="accuracy")
    if model_order:
        global_accuracy = global_accuracy.reindex(
            [m for m in model_order if m in global_accuracy.index]
        )
    global_accuracy.to_csv(tables_dir / "confronto_globale_accuratezza.csv")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison(global_accuracy, figures_dir / "confronto_globale_accuratezza.png")
    return global_accuracy


def update_degradation_figure(combined_ideal: pd.DataFrame, tables_dir: Path,
                               figures_dir: Path) -> bool:
    """Rigenera la Figura `degrado_performance` (sezione "Effetto del rumore e del
    passaggio a hardware reale", Capitolo 4) a partire dai risultati ideali già aggregati
    (`combined_ideal`, con colonne `dataset`, `model`, `accuracy`; si veda
    `aggregate_ideal_results`) e, se presenti in `tables_dir`, dalle tabelle
    `results_noisy_simulation.csv`/`results_real_hardware.csv` prodotte dal notebook
    `06_hardware_execution.ipynb`. Le due tabelle di rumore/hardware sono opzionali: se
    assenti (campagna sperimentale non ancora eseguita in quelle modalità), la figura viene
    comunque generata con il solo ambiente ideale, così da poter essere aggiornata in modo
    incrementale. Restituisce `True` se la figura è stata generata (almeno una riga QSVC/VQC
    disponibile), `False` altrimenti (nessun risultato QSVC/VQC in `combined_ideal`).

    Riutilizzata identicamente da `run_pipeline.py --aggregate` e dal notebook
    `07_results_comparison.ipynb` (si veda `aggregate_ideal_results`).
    """
    rows = [
        {"dataset": r["dataset"], "model": r["model"], "environment": "ideal",
         "accuracy": r["accuracy"]}
        for r in combined_ideal.to_dict("records") if r["model"] in ("QSVC", "VQC")
    ]

    for environment, filename in (
        ("noisy_simulation", "results_noisy_simulation.csv"),
        ("real_hardware", "results_real_hardware.csv"),
    ):
        path = tables_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows.extend({
            "dataset": r["dataset"], "model": r["model"], "environment": environment,
            "accuracy": r["accuracy"],
        } for r in df.to_dict("records"))

    if not rows:
        return False
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_noise_hardware_degradation(rows, figures_dir / "degrado_performance_ambienti.png")
    return True
