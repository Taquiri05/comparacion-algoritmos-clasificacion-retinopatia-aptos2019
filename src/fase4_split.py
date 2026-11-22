# src\fase4_split.py
# FASE 4 - PASO 1: division 80/20 estratificada (random_state=42). El test no se vuelve a tocar hasta la Fase 6.
import os, pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(RAIZ, "datos", "caracteristicas.csv"))

train, test = train_test_split(df, test_size=0.20, stratify=df["y"], random_state=42)

# Comprobaciones: ninguna foto en ambos grupos y no se pierde ninguna fila
assert set(train["id_code"]).isdisjoint(set(test["id_code"])), "hay fotos repetidas entre train y test"
assert len(train) + len(test) == len(df), "se perdieron filas"

train.to_csv(os.path.join(RAIZ, "datos", "train.csv"), index=False)
test.to_csv(os.path.join(RAIZ, "datos", "test.csv"), index=False)

print(f"Total: {len(df)} | train: {len(train)} | test: {len(test)}")
for nombre, d in [("total", df), ("train", train), ("test", test)]:
    print(f"{nombre:6s} sin RD: {(d['y']==0).sum():5d} | con RD: {(d['y']==1).sum():5d} | % con RD: {d['y'].mean()*100:.2f}")
print("\nConteo por nivel (0 a 4):")
print(pd.DataFrame({"train": train["diagnosis"].value_counts().sort_index(),
                    "test": test["diagnosis"].value_counts().sort_index()}))
print("\nGuardados: datos\\train.csv y datos\\test.csv")
