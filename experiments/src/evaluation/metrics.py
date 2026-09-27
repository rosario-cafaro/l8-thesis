"""Metriche di valutazione, comuni a modelli classici e quantistici
(Sezione "Modulo di valutazione", Capitolo 4)."""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import time

import numpy as np


def evaluate_model(y_true, y_pred, average: str = "macro"):
    # zero_division=0: se un modello (tipicamente il VQC, in caso di
    # scarsa convergenza) non predice mai una classe, la precisione/il
    # richiamo per quella classe sono indefiniti (0/0); si sceglie
    # esplicitamente 0.0.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def timed_fit_predict(model, X_train, y_train, X_test):
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit

    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    return y_pred, fit_time, predict_time


def circuit_stats(circuit) -> dict:
    """Profondità logica di un circuito quantistico (feature map, circuito
    di fedeltà del QSVC o feature map + ansatz del VQC). La profondità
    dopo la transpilazione sulle porte native di un dispositivo è calcolata da
    `transpiled_circuit_stats`."""
    decomposed = circuit.decompose()
    return {
        "depth": decomposed.depth(),
    }


def transpiled_circuit_stats(circuit, pass_manager) -> dict:
    """Profondità e numero di porte a due qubit del circuito dopo la
    transpilazione con `pass_manager` (si veda
    `src.execution.backend_manager.get_pass_manager`), cioè del circuito
    effettivamente eseguito sulle porte native del dispositivo, incluse
    le eventuali porte SWAP inserite dal routing: sono queste, e non la
    profondità logica di `circuit_stats`, le grandezze rilevanti per
    l'esposizione al rumore."""
    measured = circuit.copy()
    measured.measure_all()
    transpiled = pass_manager.run(measured)
    n_2q = sum(1 for instruction in transpiled.data
               if instruction.operation.num_qubits == 2)
    return {"transpiled_depth": transpiled.depth(), "n_2q_gates": n_2q}


def kernel_offdiag_stats(feature_map, X) -> dict:
    """Statistiche degli elementi fuori diagonale della matrice di kernel
    a fedeltà K(x, x') = |<psi(x)|psi(x')>|^2 sui campioni `X`, calcolata
    esattamente sugli statevector (senza campionamento a shot finiti).

    Il riferimento `haar_reference` = 1/2^n è il valore atteso della
    fedeltà tra due stati casuali di n qubit: elementi fuori diagonale
    vicini a questo valore indicano un kernel concentrato, prossimo alla
    matrice identità, per cui ogni campione risulta quasi ortogonale a
    tutti gli altri (si veda la sezione "Limiti dello studio e minacce
    alla validità", Capitolo 4)."""
    from qiskit.quantum_info import Statevector

    states = np.array([Statevector(feature_map.assign_parameters(x)).data for x in X])
    kernel = np.abs(states.conj() @ states.T) ** 2
    off_diagonal = kernel[~np.eye(len(X), dtype=bool)]
    return {
        "kernel_offdiag_mean": float(off_diagonal.mean()),
        "kernel_offdiag_median": float(np.median(off_diagonal)),
        "kernel_offdiag_std": float(off_diagonal.std()),
        "haar_reference": 2.0 ** -feature_map.num_qubits,
    }
