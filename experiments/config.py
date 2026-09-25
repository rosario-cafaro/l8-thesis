"""Configurazione condivisa da notebook e script (sezione "Configurazione
sperimentale", Tabella "Configurazione sperimentale completa").

Centralizza gli iperparametri specifici di ciascun dataset (numero di
componenti PCA / qubit, ripetizioni della feature map e dell'ansatz).
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Numero massimo di iterazioni dell'ottimizzatore COBYLA per il VQC
# (sezione "Configurazione sperimentale").
VQC_MAXITER = 100

DATASET_CONFIGS = {
    "breast_cancer": {
        "n_components": 4,
        "feature_map_reps": 2,
        "ansatz_reps": 3,
        "entanglement": "linear",
    },
    "wine": {
        "n_components": 5,
        "feature_map_reps": 2,
        "ansatz_reps": 3,
        "entanglement": "linear",
    },
    "digits": {
        "n_components": 6,
        "feature_map_reps": 2,
        "ansatz_reps": 3,
        "entanglement": "linear",
    },
}

# Valori aggiuntivi di componenti PCA utilizzati nell'analisi di
# sensitività del dataset Digits (sezione "Effetto della riduzione
# dimensionale", Tabella
# "digits_sensitivity_pca").
DIGITS_PCA_SENSITIVITY_COMPONENTS = [4, 5, 6, 8]

# Sottocampionamento stratificato di training/test set per le modalità
# noisy_simulation e real_hardware (sezione "Limiti dello studio e
# minacce alla validità"). Usato da notebooks/06_hardware_execution.ipynb;
# centralizzato qui, come gli altri iperparametri sperimentali, anziché
# definito localmente nel notebook.
HARDWARE_TRAIN_SAMPLE_SIZE = 60
HARDWARE_TEST_SAMPLE_SIZE = 20

# Dimensione dei chunk per il calcolo della matrice di kernel del QSVC in
# modalità noisy_simulation (sezione "Modello QSVC"): il vincolo qui è il
# picco di memoria locale di AerSampler, che ha un'impronta per coppia
# sostanzialmente più ripida di StatevectorSampler (si veda la sezione
# "Limiti dello studio e minacce alla validità").
QSVC_MAX_CIRCUITS_PER_JOB = 100

# Dimensione dei chunk per il calcolo della matrice di kernel del QSVC in
# modalità real_hardware, distinta da QSVC_MAX_CIRCUITS_PER_JOB perché qui
# il vincolo non è la memoria locale (l'esecuzione dei circuiti avviene sui
# server IBM, non in locale) ma il costo/tempo per singolo job: individuato
# empiricamente un overhead fisso per job dell'ordine del minuto,
# indipendente dal numero di circuiti in esso contenuti (fino al limite di
# servizio di 10 milioni di "esecuzioni", circuiti per shot, per job), per
# cui pochi job più grandi sono preferibili a molti job piccoli. Il valore
# è ancora sperimentale (verificato empiricamente solo a 100 circuiti/job).
QSVC_REAL_HARDWARE_MAX_CIRCUITS_PER_JOB = 1000

# Dimensione dei chunk per il calcolo della matrice di kernel del QSVC in
# simulazione ideale (StatevectorSampler), distinta da QSVC_MAX_CIRCUITS_PER_JOB
# perché qui i training set non sono sottocampionati (fino a circa 1.400
# campioni per Digits, contro i 60 di HARDWARE_TRAIN_SAMPLE_SIZE): un valore
# piccolo come 100 genererebbe decine di migliaia di chunk, con un overhead
# complessivo non trascurabile; individuato empiricamente (si veda la sezione
# "Limiti dello studio e minacce alla validità") come compromesso tra un
# picco di memoria contenuto per FidelityQuantumKernel (circa 7,5KB per
# coppia di campioni: circa 375MB per chunk a questa dimensione) e un
# numero di chunk moderato (poche decine anche per Digits).
QSVC_IDEAL_MAX_CIRCUITS_PER_JOB = 50_000

DATASET_NAMES = list(DATASET_CONFIGS.keys())

# Seed multipli per la ripetizione dell'addestramento del VQC in
# simulazione ideale (sezione "Variabilità del VQC su seed multipli"),
# a partire da RANDOM_STATE per includere, tra le ripetizioni, lo stesso
# seed usato nel resto della pipeline: un solo seed garantisce la
# riproducibilità dell'esperimento ma non consente di valutarne
# adeguatamente la variabilità.
VQC_SEED_REPEATS = [RANDOM_STATE + i for i in range(5)]
