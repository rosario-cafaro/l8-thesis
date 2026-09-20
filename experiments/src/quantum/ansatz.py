"""Costruzione dell'ansatz variazionale utilizzato dal VQC (Sezione "Modello
VQC", Capitolo 4)."""

from qiskit.circuit.library import real_amplitudes


def build_ansatz(n_qubits: int, reps: int = 3,
                  entanglement: str = "linear"):
    """Costruisce l'ansatz variazionale RealAmplitudes.

    Il numero di qubit deve coincidere con quello della feature map
    a cui l'ansatz viene applicato in sequenza (si veda
    src.quantum.feature_maps.build_feature_map).

    Usa la funzione `real_amplitudes` anziché la classe `RealAmplitudes`
    (deprecata a partire da Qiskit 2.1, rimossa in Qiskit 3.0).
    """
    return real_amplitudes(
        num_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
    )
