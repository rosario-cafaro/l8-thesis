"""Modelli classici di baseline (Sezione "Modelli classici di baseline",
Capitolo 4)."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Spazio di ricerca degli iperparametri utilizzato dalla grid search
# con validazione incrociata stratificata a 5 fold (si veda
# src.pipeline.train_classical_models).
GRID_SEARCH_PARAMS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF kernel)": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
    },
}


def get_classical_models(random_state: int = 42):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        # random_state non viene passato a SVC: scikit-learn lo utilizza
        # esclusivamente per la stima di probabilita' (`probability=True`,
        # non usata qui), quindi sarebbe un parametro morto che non
        # influenzerebbe in alcun modo il modello (RBF-SVC e' comunque
        # deterministico in questa configurazione).
        "SVM (RBF kernel)": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
    }
