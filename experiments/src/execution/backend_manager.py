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


def _select_qubit_path(coupling_map, n_qubits: int, start: int = 0) -> list[int]:
    """Seleziona, tramite ricerca in profondità sulla coupling map del
    backend a partire dal qubit fisico `start`, un cammino semplice di
    `n_qubits` qubit fisici, in cui ciascun qubit è collegato al
    successivo; usato come layout iniziale del circuito (si veda
    `get_pass_manager`).

    Il cammino (e non un generico sottoinsieme connesso, come quello
    restituito da una visita in ampiezza) rispecchia lo schema di
    entanglement `linear` di feature map e ansatz, che agisce sulle
    coppie di qubit logici (i, i+1): mappandole su qubit fisici
    adiacenti, il transpiler non deve inserire porte SWAP."""
    adjacency: dict[int, set[int]] = {}
    for a, b in coupling_map.get_edges():
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    def extend(path: list[int]) -> list[int] | None:
        if len(path) == n_qubits:
            return path
        for neighbor in sorted(adjacency.get(path[-1], ())):
            if neighbor not in path:
                found = extend(path + [neighbor])
                if found is not None:
                    return found
        return None

    path = extend([start])
    if path is None:
        raise ValueError(
            f"Impossibile trovare un cammino di {n_qubits} qubit fisici "
            f"adiacenti a partire dal qubit {start} sul backend indicato."
        )
    return path


def get_pass_manager(mode: str, backend=None, n_qubits: int = None,
                      optimization_level: int = 1, seed: int = None):
    """Costruisce il pass manager di transpilazione necessario per le
    modalità `noisy_simulation` e `real_hardware`: il `SamplerV2` di
    Qiskit Runtime accetta solo circuiti già transpilati sul target del
    dispositivo, e il modello di rumore di `AerSampler`
    associa ciascun canale di errore alla porta nativa e agli
    indici dei qubit fisici su cui agisce. Restituisce `None` per la
    modalità `ideal`, che non ne ha bisogno.

    Il circuito è transpilato contro l'intero backend, con layout
    iniziale fissato su un cammino di `n_qubits` qubit fisici adiacenti
    (si veda `_select_qubit_path`): gli indici dei qubit del circuito
    transpilato sono quindi quelli fisici del dispositivo, gli stessi
    con cui sono indicizzati il target dell'hardware reale e il modello
    di rumore costruito da `build_noise_model_from_backend`.

    Il circuito transpilato è largo quanto l'intero registro del
    dispositivo, ma i qubit inattivi non ricevono né porte né misure
    (`measure_all` è applicato dalle classi di qiskit-machine-learning
    prima della transpilazione, sul solo circuito logico), e Aer li
    rimuove prima della simulazione (troncamento dei qubit inattivi,
    attivo per default): il costo della simulazione dipende quindi solo
    da `n_qubits`.

    `seed` fissa il generatore pseudocasuale del transpiler
    (`seed_transpiler`), che alcune fasi (ad esempio il routing) usano
    anche con layout iniziale fissato: senza di esso due transpilazioni
    dello stesso circuito possono differire, e con esse i risultati della
    simulazione con rumore.
    """
    if mode == "ideal":
        return None
    if backend is None or n_qubits is None:
        raise ValueError(
            f"Sono necessari sia backend sia n_qubits per costruire il "
            f"pass manager in modalità '{mode}'."
        )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    physical_qubits = _select_qubit_path(backend.coupling_map, n_qubits)
    return generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        initial_layout=physical_qubits,
        seed_transpiler=seed,
    )
