"""Costruzione della feature map quantistica (Sezione "Codifica quantistica:
feature map", Capitolo 7)."""

from qiskit.circuit.library import zz_feature_map


def build_feature_map(n_qubits: int, reps: int = 2,
                       entanglement: str = "linear"):
    """Costruisce la feature map utilizzata per la codifica dei dati.

    Il numero di qubit coincide con il numero di componenti
    principali selezionate in fase di preprocessing
    (si veda la sezione "Riduzione della dimensionalità tramite PCA",
    Capitolo 6).

    Usa la funzione `zz_feature_map` anziché la classe `ZZFeatureMap`
    (deprecata a partire da Qiskit 2.1, rimossa in Qiskit 3.0).
    """
    feature_map = zz_feature_map(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement=entanglement,
    )
    return feature_map
