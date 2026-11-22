# Figura de la Fase 1 con las 3,530 imagenes unicas (no abre imagenes; solo lee CSV de datos/)
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOMBRES_NIVEL = ["diagnosis", "nivel"]
partes = []
for ruta in ["datos/train.csv", "datos/test.csv"]:
    df = pd.read_csv(ruta)
    col = next((c for c in NOMBRES_NIVEL if c in df.columns), None)
    if "id_code" in df.columns and col:
        partes.append(df[["id_code", col]].rename(columns={col: "nivel"}).assign(origen=ruta))
if not partes:
    print("No encontre una columna de nivel (diagnosis/nivel) junto a id_code en datos/*.csv.")
    for ruta in ["datos/train.csv", "datos/test.csv"]:
        print(ruta, list(pd.read_csv(ruta, nrows=0).columns))
    raise SystemExit

todo = pd.concat(partes)
# Un archivo con todas las imagenes unicas, o train + test: se quitan repetidos por id_code
uni = todo.drop_duplicates("id_code")
print("Imagenes unicas:", len(uni))
if len(uni) != 3530:
    print("OJO: no da 3,530. Archivos usados:", sorted(todo["origen"].unique()), "- no se guardo la figura.")
    raise SystemExit

n = uni["nivel"].value_counts().sort_index()
sin = int((uni["nivel"] == 0).sum()); con = len(uni) - sin

fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [3, 2]})
b = axs[0].bar([str(i) for i in n.index], n.values, color="#3182bd")
for r, v in zip(b, n.values):
    axs[0].text(r.get_x() + r.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
axs[0].set_xlabel("Nivel de retinopatia (0 = sin RD, 4 = proliferativa)"); axs[0].set_ylabel("Imagenes")
axs[0].set_title("Cinco niveles (3,530 imagenes unicas)", fontsize=10); axs[0].set_ylim(0, n.max() * 1.12)
b2 = axs[1].bar(["Sin RD\n(nivel 0)", "Con RD\n(niveles 1-4)"], [sin, con], color=["#31a354", "#de2d26"])
for r, v in zip(b2, [sin, con]):
    axs[1].text(r.get_x() + r.get_width() / 2, v, f"{v:,}\n({v / len(uni) * 100:.1f} %)", ha="center", va="bottom", fontsize=9)
axs[1].set_title("Tarea binaria", fontsize=10); axs[1].set_ylim(0, max(sin, con) * 1.2)
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figuras/fig_fase1_distribucion_clases.png", dpi=300); plt.close(fig)
print("Niveles:", n.to_dict(), "| sin RD:", sin, "| con RD:", con)
print("Guardado: figuras/fig_fase1_distribucion_clases.png")

