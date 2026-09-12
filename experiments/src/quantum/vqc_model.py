"""Modello VQC: feature map + ansatz variazionale (Sezione "Modello VQC",
Capitolo 7)."""

from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.utils import algorithm_globals


def build_vqc(feature_map, ansatz, sampler, maxiter: int = 100, callback=None,
              pass_manager=None, random_state: int = None):
    """Costruisce il VQC. Non passando esplicitamente `initial_point`,
    qiskit-machine-learning genera il punto iniziale dei parametri
    variazionali estraendolo casualmente da `algorithm_globals.random`
    al momento del fit; `random_state`, se fornito, fissa il seed di
    questo generatore globale prima della costruzione, in modo che
    l'inizializzazione (e quindi l'intera traiettoria di ottimizzazione)
    sia riproducibile (Sezione "Riproducibilità degli esperimenti",
    Capitolo 6).
    """
    if random_state is not None:
        algorithm_globals.random_seed = random_state
    optimizer = COBYLA(maxiter=maxiter)
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
        pass_manager=pass_manager,
    )
    return vqc
