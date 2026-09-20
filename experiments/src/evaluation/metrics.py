"""Metriche di valutazione, comuni a modelli classici e quantistici
(Sezione "Modulo di valutazione", Capitolo 4)."""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import time


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
    """Profondità di un circuito quantistico (feature map o feature map
    + ansatz), calcolata dopo la decomposizione in porte elementari."""
    decomposed = circuit.decompose()
    return {
        "depth": decomposed.depth(),
    }
