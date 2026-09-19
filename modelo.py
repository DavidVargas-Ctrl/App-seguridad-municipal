"""
Pipeline de datos y modelo para la app de consulta municipal.

Replica exactamente la preparacion de la Parte 4 del cuaderno
(`notebooks/Proyecto_Completo_Colab.ipynb`): mismas features, mismo corte temporal,
mismos cortes de clase calculados solo con datos de entrenamiento.

Se reentrena al arrancar en vez de cargar un modelo serializado. Entrenar un Random
Forest sobre 5.565 filas tarda menos de dos segundos, y asi la app no depende de que
la version de scikit-learn del servidor coincida con la que genero el pickle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

DATOS = Path(__file__).parent / "datos"

ANIO_INICIO = 2016
ANIO_CORTE_TRAIN = 2021
RANDOM_STATE = 42
K_CLUSTERS = 4

FEATURES = [
    "gasto_per_capita_log", "gasto_per_capita_lag1_log", "gasto_vigilancia_per_capita_log",
    "share_vigilancia", "tasa_competencia_promedio", "tuvo_secop",
    "hurto_personas_per_capita_log", "hurto_vehiculos_per_capita_log",
    "estupefacientes_per_capita_log", "poblacion_log",
]

# Nombres legibles para un usuario no tecnico. La app nunca muestra el nombre interno.
NOMBRES_LEGIBLES = {
    "gasto_per_capita_log": "Inversión en seguridad por habitante",
    "gasto_per_capita_lag1_log": "Inversión del año anterior",
    "gasto_vigilancia_per_capita_log": "Inversión específica en vigilancia armada",
    "share_vigilancia": "Porcentaje del gasto dedicado a vigilancia armada",
    "tasa_competencia_promedio": "Competencia en los procesos de contratación",
    "tuvo_secop": "Tiene contratación registrada en SECOP",
    "hurto_personas_per_capita_log": "Nivel actual de hurto a personas",
    "hurto_vehiculos_per_capita_log": "Nivel actual de hurto de vehículos",
    "estupefacientes_per_capita_log": "Operativos de incautación de droga",
    "poblacion_log": "Tamaño del municipio",
}

FEATURES_CLUSTER = [
    "gasto_pc_prom_log", "gasto_pc_tend_slog", "hurto_pc_prom_log",
    "hurto_pc_tend_slog", "hurto_veh_pc_prom_log", "tasa_competencia_prom",
    "poblacion_prom_log",
]


def _pendiente(anios, valores):
    mask = ~np.isnan(valores)
    if mask.sum() < 2:
        return np.nan
    return stats.linregress(anios[mask], valores[mask])[0]


def _signed_log1p(s):
    return np.sign(s) * np.log1p(np.abs(s))


def cargar_datos():
    """Carga el panel municipio-anio y la tabla de nombres de municipio."""
    df = pd.read_csv(DATOS / "dataset_municipio_anio_final.csv", dtype={"divipola5": str})
    df["divipola5"] = df["divipola5"].str.zfill(5)
    nombres = pd.read_csv(DATOS / "municipios.csv", dtype={"divipola5": str})
    nombres["divipola5"] = nombres["divipola5"].str.zfill(5)
    return df[df["anio"] >= ANIO_INICIO].copy(), nombres


def construir_perfiles(dm, nombres):
    """Un registro por municipio: nivel y tendencia de gasto y hurto + cuadrante + cluster."""
    filas = []
    for div, g in dm.groupby("divipola5"):
        g = g.sort_values("anio")
        a = g["anio"].values.astype(float)
        filas.append({
            "divipola5": div,
            "gasto_pc_prom": g["gasto_per_capita"].mean(),
            "gasto_pc_tend": _pendiente(a, g["gasto_per_capita"].values),
            "hurto_pc_prom": g["hurto_personas_per_capita"].mean(),
            "hurto_pc_tend": _pendiente(a, g["hurto_personas_per_capita"].values),
            "hurto_veh_pc_prom": g["hurto_vehiculos_per_capita"].mean(),
            "tasa_competencia_prom": g["tasa_competencia_promedio"].mean(),
            "poblacion_prom": g["poblacion"].mean(),
            "anios_con_secop": int(g["tuvo_secop"].sum()),
            "anios_totales": len(g),
        })
    p = pd.DataFrame(filas)
    p["tasa_competencia_prom"] = p["tasa_competencia_prom"].fillna(p["tasa_competencia_prom"].median())
    p = p.dropna(subset=["gasto_pc_prom", "hurto_pc_prom", "poblacion_prom"]).reset_index(drop=True)

    # --- Cuadrantes (linea base descriptiva, seccion II.5) ---
    med_gasto, med_hurto = p["gasto_pc_prom"].median(), p["hurto_pc_prom"].median()
    p["cuadrante"] = np.select(
        [
            (p["gasto_pc_prom"] < med_gasto) & (p["hurto_pc_prom"] >= med_hurto),
            (p["gasto_pc_prom"] >= med_gasto) & (p["hurto_pc_prom"] < med_hurto),
            (p["gasto_pc_prom"] >= med_gasto) & (p["hurto_pc_prom"] >= med_hurto),
        ],
        ["bajo gasto / alto hurto", "alto gasto / bajo hurto", "alto gasto / alto hurto"],
        default="bajo gasto / bajo hurto")

    # --- Clustering (seccion II.7) ---
    p["gasto_pc_prom_log"] = np.log1p(p["gasto_pc_prom"])
    p["gasto_pc_tend_slog"] = _signed_log1p(p["gasto_pc_tend"])
    p["hurto_pc_prom_log"] = np.log1p(p["hurto_pc_prom"])
    p["hurto_pc_tend_slog"] = _signed_log1p(p["hurto_pc_tend"])
    p["hurto_veh_pc_prom_log"] = np.log1p(p["hurto_veh_pc_prom"])
    p["poblacion_prom_log"] = np.log1p(p["poblacion_prom"])
    Xc = p[FEATURES_CLUSTER].fillna(p[FEATURES_CLUSTER].median())
    p["cluster"] = KMeans(K_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(Xc))

    p["etiquetas_cluster"] = p["cluster"].map(_etiquetar_clusters(p))
    p = p.merge(nombres, on="divipola5", how="left")
    # La tabla canonica se construye a partir de los datos de la Policia, asi que le faltan los
    # municipios sin casos registrados en hurto de vehiculos ni estupefacientes. Se muestran por
    # su codigo en vez de dejar un "nan" en el selector.
    p["municipio"] = p["municipio"].fillna("Municipio " + p["divipola5"])
    p["departamento"] = p["departamento"].fillna("(departamento no identificado)")
    return p


def _etiquetar_clusters(p):
    """Etiqueta cada cluster por la posicion real de sus municipios (ver seccion II.7).

    El umbral de "gasto alto" se calcula solo entre municipios CON dato de gasto: 485 de
    1.110 no tienen ningun proceso SECOP registrado, y usar la mediana de todos haria que
    "gasto alto" significara apenas "gasto algo".
    """
    con_dato = p["anios_con_secop"] > 0
    u_gasto = p.loc[con_dato, "gasto_pc_prom"].median()
    u_hurto = p["hurto_pc_prom"].median()
    diag = p.assign(
        ga=p["gasto_pc_prom"] >= u_gasto,
        ha=p["hurto_pc_prom"] >= u_hurto,
        sd=p["anios_con_secop"] == 0,
    ).groupby("cluster").agg(sd=("sd", "mean"), ga=("ga", "mean"), ha=("ha", "mean"),
                             gm=("gasto_pc_prom", "median"))
    base = {}
    for cl in diag.index:
        if diag.loc[cl, "sd"] > 0.5:
            base[cl] = "Sin información de gasto"
        elif diag.loc[cl, "ga"] > 0.5 and diag.loc[cl, "ha"] <= 0.5:
            base[cl] = "Alto gasto, bajo hurto"
        elif diag.loc[cl, "ga"] <= 0.5 and diag.loc[cl, "ha"] > 0.5:
            base[cl] = "Bajo gasto, alto hurto"
        elif diag.loc[cl, "ga"] > 0.5:
            base[cl] = "Alto gasto, alto hurto (gasto reactivo)"
        else:
            base[cl] = "Gasto y hurto bajos"
    etiquetas = dict(base)
    for etq in {e for e in base.values() if list(base.values()).count(e) > 1}:
        cls = sorted([c for c, e in base.items() if e == etq], key=lambda c: -diag.loc[c, "gm"])
        for pos, c in enumerate(cls):
            etiquetas[c] = f"{etq} — {['intensidad muy alta', 'intensidad alta', 'intensidad media'][min(pos, 2)]}"
    return etiquetas


def preparar_modelado(dm):
    """Construye el target (cambio del hurto a t+1) y las features. Ver seccion II.6."""
    d = dm.sort_values(["divipola5", "anio"]).copy()
    d["hpp_siguiente"] = d.groupby("divipola5")["hurto_personas_per_capita"].shift(-1)
    d["anio_siguiente"] = d.groupby("divipola5")["anio"].shift(-1)
    d["cambio_hurto"] = d["hpp_siguiente"] - d["hurto_personas_per_capita"]
    d["gasto_per_capita_lag1"] = d.groupby("divipola5")["gasto_per_capita"].shift(1).fillna(0)

    entrenable = d[(d["anio_siguiente"] == d["anio"] + 1)
                   & d["hurto_personas_per_capita"].notna() & d["hpp_siguiente"].notna()].copy()

    # Cortes de clase calculados SOLO con anios de entrenamiento: usarlos sobre todo el panel
    # seria filtrar informacion del futuro hacia la definicion misma del target.
    es_train = entrenable["anio"] <= ANIO_CORTE_TRAIN
    cortes = entrenable.loc[es_train, "cambio_hurto"].quantile([1 / 3, 2 / 3]).values
    entrenable["clase"] = pd.cut(entrenable["cambio_hurto"],
                                 [-np.inf, cortes[0], cortes[1], np.inf],
                                 labels=["mejoro", "se_mantuvo", "empeoro"])

    # Las filas del ultimo anio no tienen target, pero si sirven para predecir hacia adelante.
    ultimo = d.groupby("divipola5").tail(1).copy()
    ultimo["clase"] = None

    for frame in (entrenable, ultimo):
        for col in ["tasa_competencia_promedio", "share_vigilancia"]:
            frame[col] = frame[col].fillna(entrenable.loc[es_train, col].median())
        for c in ["gasto_per_capita", "gasto_per_capita_lag1", "gasto_vigilancia_per_capita",
                  "poblacion", "hurto_personas_per_capita", "hurto_vehiculos_per_capita",
                  "estupefacientes_per_capita"]:
            frame[c + "_log"] = np.log1p(frame[c].clip(lower=0))
    return entrenable, ultimo


def entrenar(entrenable):
    """Random Forest con los hiperparametros que gano la busqueda del cuaderno."""
    train = entrenable[entrenable["anio"] <= ANIO_CORTE_TRAIN]
    test = entrenable[entrenable["anio"] > ANIO_CORTE_TRAIN]
    modelo = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    modelo.fit(train[FEATURES], train["clase"].astype(str))

    pred_test = modelo.predict(test[FEATURES])
    real_test = test["clase"].astype(str).values
    marcados = pred_test == "empeoro"
    metricas = {
        "precision_alerta": float((real_test[marcados] == "empeoro").mean()) if marcados.any() else 0.0,
        "n_marcados": int(marcados.sum()),
        "accuracy": float((pred_test == real_test).mean()),
    }
    return modelo, metricas
