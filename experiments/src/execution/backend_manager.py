"""Gestione dei backend di esecuzione: simulatore ideale, simulatore con
rumore e hardware quantistico reale (Sezione "Gestione dei backend di
esecuzione", Capitolo 4)."""

import os

from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers import SamplerPub
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.noise import NoiseModel


class CountingSampler:
    """Wrapper attorno a un sampler Qiskit che conta il numero di singoli
    circuiti effettivamente inviati per l'esecuzione (una combinazione di
    valori numerici legata al circuito conta come un circuito eseguito),
    utile per popolare la colonna "N. circuiti eseguiti" della Tabella
    costo_computazionale (Sezione "Costo computazionale e tempi di
    esecuzione", Capitolo 4). Delega ogni altra operazione al
    sampler avvolto, di cui replica l'interfaccia (`run`)."""

    def __init__(self, sampler):
        self._sampler = sampler
        self.n_circuits = 0

    def run(self, pubs, *args, **kwargs):
        pubs = list(pubs)
        for pub in pubs:
            self.n_circuits += SamplerPub.coerce(pub).parameter_values.size
        return self._sampler.run(pubs, *args, **kwargs)


def get_sampler(mode: str, noise_model: NoiseModel = None,
                 backend_name: str = None, seed: int = None):
    """Costruisce il sampler per la modalità richiesta.

    `StatevectorSampler` e `AerSampler` non calcolano probabilità
    esatte per default, ma campionano un numero finito di shot
    (`default_shots=1024`) da un generatore pseudocasuale interno; se
    `seed` non viene fissato, tale generatore non è seedato e due
    esecuzioni identiche restituiscono stime del kernel/della funzione
    di costo diverse, compromettendo la riproducibilità (Sezione
    "Riproducibilità degli esperimenti", Capitolo 3). Non è invece
    possibile fissare un seed per l'hardware reale, per sua natura
    non deterministico.
    """
    if mode == "ideal":
        return StatevectorSampler(seed=seed)

    elif mode == "noisy_simulation":
        return AerSampler(seed=seed, options={"backend_options":
                                               {"noise_model": noise_model}})

    elif mode == "real_hardware":
        # Importato localmente per non introdurre una dipendenza rigida
        # da qiskit-ibm-runtime (e dalle relative credenziali) nei
        # percorsi di esecuzione ideale e con rumore simulato.
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        # Token e channel vengono letti dalle variabili d'ambiente
        # QISKIT_IBM_TOKEN/QISKIT_IBM_CHANNEL dal file .env,
        # per coerenza con quanto descritto nella Sezione "Struttura del
        # progetto e ambiente containerizzato" (Capitolo 4).
        service = QiskitRuntimeService(
            channel=os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
            token=os.environ.get("QISKIT_IBM_TOKEN"),
        )
        backend = service.backend(backend_name)
        return SamplerV2(mode=backend)

    else:
        raise ValueError(f"Modalità di esecuzione non valida: {mode}")


def build_noise_model_from_backend(backend) -> NoiseModel:
    """Costruisce un modello di rumore Aer a partire dai dati di
    calibrazione pubblicati per un backend reale IBM, da utilizzare
    con `get_sampler(mode="noisy_simulation", ...)` senza dover
    effettivamente eseguire i circuiti sull'hardware.
    """
    return NoiseModel.from_backend(backend)


def _select_connected_qubits(coupling_map, n_qubits: int) -> list[int]:
    """Seleziona, tramite visita in ampiezza sulla coupling map del
    backend a partire dal qubit fisico 0, un sottoinsieme di `n_qubits`
    qubit fisici mutuamente connessi. Necessario perché transpilare
    contro l'intero backend produrrebbe un circuito largo quanto il suo
    registro completo (si veda `get_pass_manager`), anche se il
    circuito logico ne usa solo una piccola parte."""
    from collections import deque

    adjacency: dict[int, set[int]] = {}
    for a, b in coupling_map.get_edges():
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    visited: list[int] = []
    seen = {0}
    queue = deque([0])
    while queue and len(visited) < n_qubits:
        qubit = queue.popleft()
        visited.append(qubit)
        for neighbor in sorted(adjacency.get(qubit, ())):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    if len(visited) < n_qubits:
        raise ValueError(
            f"Impossibile trovare {n_qubits} qubit fisici mutuamente connessi "
            f"a partire dal qubit 0 sul backend indicato."
        )
    return visited[:n_qubits]


def get_pass_manager(mode: str, backend=None, n_qubits: int = None,
                      optimization_level: int = 1):
    """Costruisce il pass manager di transpilazione necessario per le
    modalità `noisy_simulation` e `real_hardware`: a differenza di
    `StatevectorSampler` (modalità `ideal`), `AerSampler` e il
    `SamplerV2` di Qiskit Runtime richiedono circuiti già espressi nel
    basis gate set del backend/target (transpilati), e non accettano
    direttamente le porte di libreria (ad esempio `ZZFeatureMap`)
    utilizzate da `build_feature_map`/`build_ansatz`. Restituisce
    `None` per la modalità `ideal`, che non ne ha bisogno.

    Il pass manager viene costruito contro una coupling map ridotta a
    `n_qubits` qubit fisici mutuamente connessi (si veda
    `_select_connected_qubits`), anziché contro l'intero backend: i
    dispositivi IBM attualmente disponibili hanno tipicamente 127 o più
    qubit, mentre questo lavoro ne usa 4--6 (Tabella
    `configurazione_completa`); transpilare contro l'intero backend
    produce un circuito largo quanto il suo registro completo (i qubit
    inattivi restano comunque dichiarati nel circuito), rendendo la
    simulazione con rumore su `AerSampler` computazionalmente
    infattibile.
    """
    if mode == "ideal":
        return None
    if backend is None or n_qubits is None:
        raise ValueError(
            f"Sono necessari sia backend sia n_qubits per costruire il "
            f"pass manager in modalità '{mode}'."
        )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

    physical_qubits = _select_connected_qubits(backend.coupling_map, n_qubits)
    # .reduce() rinumera i qubit fisici selezionati in un intervallo
    # compatto 0..n_qubits-1: senza questa rinumerazione il circuito
    # transpilato erediterebbe comunque gli indici originali (spesso
    # alti su un dispositivo da 127+ qubit), risultando largo quanto
    # l'indice massimo anziché quanto n_qubits.
    reduced_coupling_map = backend.coupling_map.reduce(physical_qubits)

    # A partire da Qiskit 2.1, generate_preset_pass_manager valida i nomi in
    # basis_gates contro un elenco di porte "standard" quando non riceve
    # anche l'argomento backend, sollevando un ValueError ("Providing
    # non-standard gates... is not allowed") per istruzioni non standard
    # eventualmente esposte da un backend reale (ad esempio measure_reset su
    # alcuni dispositivi IBM), riscontrato empiricamente eseguendo su
    # hardware reale. Passare backend=backend risolverebbe la validazione ma
    # farebbe costruire un target con num_qubits pari all'intero registro del
    # backend (Target.from_configuration usa backend.num_qubits
    # indipendentemente dalla coupling_map ridotta fornita, verificato
    # empiricamente), vanificando la riduzione dei qubit appena descritta. Si
    # filtra quindi basis_gates alle sole porte standard riconosciute da
    # Qiskit: tutte le porte native reali (ad esempio ecr, cx, sx, rz) sono
    # incluse in questo elenco; le eventuali istruzioni non standard scartate
    # non sono comunque necessarie ai circuiti costruiti da questo lavoro.
    standard_gate_names = set(get_standard_gate_name_mapping()) | {"measure", "delay", "reset"}
    basis_gates = [name for name in backend.operation_names if name in standard_gate_names]

    return generate_preset_pass_manager(
        optimization_level=optimization_level,
        basis_gates=basis_gates,
        coupling_map=reduced_coupling_map,
    )
