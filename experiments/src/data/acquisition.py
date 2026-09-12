"""Acquisizione e validazione preliminare dei dataset (Sezione "Acquisizione
e validazione dei dati", Capitolo 7)."""

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
import pandas as pd

_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}


def load_dataset(name: str):
    if name not in _LOADERS:
        raise ValueError(f"Dataset non supportato: {name}")

    raw = _LOADERS[name](as_frame=True)
    X, y = raw.data, raw.target
    validate_dataset(X, y)
    return X, y


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Verifica preliminare di integrità del dataset.

    Usa controlli espliciti (`if`/`raise`) anziché `assert`: a differenza
    di `assert`, non vengono rimossi dal bytecode se l'interprete gira
    con ottimizzazioni attive (`python -O`/`PYTHONOPTIMIZE=1`).
    """
    if X.isnull().sum().sum() != 0:
        raise ValueError("Valori mancanti rilevati")
    if not X.select_dtypes(exclude="number").empty:
        raise ValueError("Sono presenti feature non numeriche")
    class_counts = y.value_counts()
    if class_counts.min() / class_counts.max() < 0.3:
        print(f"Attenzione: sbilanciamento significativo tra le classi "
              f"({class_counts.to_dict()})")
