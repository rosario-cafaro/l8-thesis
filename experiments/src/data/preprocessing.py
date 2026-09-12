"""Normalizzazione Min-Max e riduzione dimensionale tramite PCA (Sezione
"Preprocessing: normalizzazione Min-Max e PCA", Capitolo 7)."""

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def preprocess(X, y, n_components: int, test_size: float = 0.2,
                random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Normalizzazione Min-Max: fit esclusivamente sul train
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Riduzione dimensionale tramite PCA: fit esclusivamente sul train
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca


def stratified_subsample(X, y, n_samples: int, random_state: int = 42):
    """Sottocampiona (X, y) a `n_samples` campioni preservando le
    proporzioni tra classi.

    Usata per ridurre la dimensione di training/test set nelle modalità
    di esecuzione più costose (`noisy_simulation`, `real_hardware`,
    Sezione "Limiti dello studio e minacce alla validità", Capitolo 7):
    un semplice troncamento posizionale (`X[:n_samples]`)
    non garantisce la stessa distribuzione di classe dell'insieme
    originario, anche quando quest'ultimo è stato mescolato da uno split
    stratificato a monte. Se `n_samples` è maggiore o uguale al numero di
    campioni disponibili, restituisce (X, y) invariati.
    """
    if n_samples >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_samples, stratify=y, random_state=random_state,
    )
    return X_sub, y_sub
