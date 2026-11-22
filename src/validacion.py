# Validacion cruzada comun a los 6 modelos (mismos pliegues para todos)
from sklearn.model_selection import StratifiedKFold

N_PLIEGUES = 5
SEMILLA = 42

def crear_cv():
    """5 pliegues estratificados, mezclados, con semilla fija."""
    return StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
