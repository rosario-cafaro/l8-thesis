"""Modello QSVC basato su kernel quantistico a fedeltà (Sezione "Modello
QSVC", Capitolo 4)."""

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def build_qsvc(feature_map, sampler, pass_manager=None, max_circuits_per_job=None):
    """Costruisce il QSVC a partire dalla feature map e dal sampler.

    Il sampler determina il backend di esecuzione dei circuiti di
    fedelta' (simulatore ideale, simulatore con rumore o hardware
    reale, si veda src.execution.backend_manager.get_sampler) ed e'
    esplicitamente collegato al kernel quantistico tramite la
    primitiva ComputeUncompute. Il pass_manager (obbligatorio per le
    modalita' diverse da quella ideale, si veda
    src.execution.backend_manager.get_pass_manager) transpila i
    circuiti nel basis gate set del backend prima dell'esecuzione.

    Per costruire la matrice di kernel, FidelityQuantumKernel invoca il
    sampler una sola volta, passando in un'unica chiamata una coppia di
    circuiti per ciascuna coppia di campioni di addestramento (una
    lista di PUB, si veda ComputeUncompute._run). Con max_circuits_per_job
    non impostato, per training set di alcune migliaia di campioni
    questo genera una singola chiamata con centinaia di migliaia di PUB
    (circa 8KB di picco di memoria per coppia, misurato empiricamente),
    che puo' esaurire la memoria disponibile prima ancora di raggiungere
    l'intero training set, non solo su AerSampler (simulazione con
    rumore), ma anche su StatevectorSampler (simulazione ideale) per
    training set sufficientemente grandi (si veda il dataset Digits,
    Sezione "Limiti dello studio e minacce alla validita'", Capitolo 4).
    Impostando max_circuits_per_job la chiamata viene suddivisa in piu'
    chiamate piu' piccole (una per "chunk" di coppie), riducendo
    drasticamente il picco di memoria per chiamata a fronte di un
    modesto overhead aggiuntivo per chunk (tempo totale pressoche'
    invariato, verificato empiricamente).
    """
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map, fidelity=fidelity,
        max_circuits_per_job=max_circuits_per_job,
    )
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    return qsvc
