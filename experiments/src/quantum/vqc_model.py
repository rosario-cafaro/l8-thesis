"""Modello VQC: feature map + ansatz variazionale (Sezione "Modello VQC",
Capitolo 4)."""

from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.utils import algorithm_globals


def build_vqc(feature_map, ansatz, sampler, num_classes: int, maxiter: int = 100,
              callback=None, pass_manager=None, random_state: int = None,
              optimizer=None):
    """Costruisce il VQC. Non passando esplicitamente `initial_point`,
    qiskit-machine-learning genera il punto iniziale dei parametri
    variazionali estraendolo casualmente da `algorithm_globals.random`
    al momento del fit; `random_state`, se fornito, fissa il seed di
    questo generatore globale prima della costruzione, in modo che
    l'inizializzazione (e quindi l'intera traiettoria di ottimizzazione)
    sia riproducibile (Sezione "Riproducibilità degli esperimenti",
    Capitolo 3).

    `num_classes` è passato a VQC come `output_shape`: senza di esso,
    qiskit-machine-learning (0.9.1) costruisce la funzione di
    interpretazione `x % 2` del caso binario e, al fit, adatta la
    dimensione dell'output al numero di classi senza aggiornare tale
    funzione, per cui le classi >= 2 riceverebbero probabilità
    identicamente nulla. Con `output_shape=num_classes` l'interpretazione
    diventa `x % num_classes`: ciascuno dei 2^n stati di base misurati è
    assegnato a una classe (nel caso binario il comportamento è identico
    al default).

    `optimizer`, se non fornito, è COBYLA con `maxiter` valutazioni della
    funzione di costo; un ottimizzatore diverso (ad esempio SPSA nel
    confronto tra ottimizzatori) va passato già configurato, e in tal
    caso `maxiter` è ignorato.
    """
    if random_state is not None:
        algorithm_globals.random_seed = random_state
    if optimizer is None:
        optimizer = COBYLA(maxiter=maxiter)
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
        pass_manager=pass_manager,
        output_shape=num_classes,
    )
    return vqc
