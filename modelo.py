"""
Pipeline de datos y modelo para la app de consulta municipal.

ACTUALIZADO a la Parte 4 rediseñada del cuaderno (`Proyecto_Completo_Colab.ipynb`).

Qué cambió respecto a la version anterior de este archivo:

1. Ya no hay UN problema (hurto a personas). Hay CINCO, resueltos en paralelo, uno por
   categoria de hurto: Personas, Vehiculos, Residencias, Comercio y el compendio
   `hurto_total`. La app deja que el usuario elija cual mirar.
2. Regla de exclusion de la seccion II.6: el modelo de una categoria NO usa las demas
   categorias de hurto como features. La unica excepcion es el nivel actual de la propia
   categoria, que se mantiene como control (sin el no se distingue "gasto reactivo" de
   "gasto que funciono").
3. Features de contexto nuevas: `homicidios_per_capita` y `capturas_per_capita` entran al
   modelo; `hurto_vehiculos_per_capita` sale (era una categoria de hurto usada para
   predecir otra categoria de hurto).
4. El algoritmo ya no es Random Forest para todo: se usa el ganador por F1 macro de cada
   categoria segun la tabla de 15 combinaciones del cuaderno (Gradient Boosting en 4 de 5,
   Random Forest en Comercio).

Se reentrena al arrancar en vez de cargar un modelo serializado, por la misma razon de
siempre: asi la app no depende de que la version de scikit-learn del servidor coincida con
la que genero el pickle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DATOS = Path(__file__).parent / "datos"

ANIO_INICIO = 2016
ANIO_CORTE_TRAIN = 2021
RANDOM_STATE = 42
K_CLUSTERS = 4          # K_NEGOCIO en el cuaderno: alto/bajo gasto x alto/bajo hurto
CLASES = ["mejoro", "se_mantuvo", "empeoro"]

# --------------------------------------------------------------------------------------
# Las 5 categorias de la seccion II.6.
#   clave -> (etiqueta visible, columna de total, columna per capita, algoritmo ganador)
# El algoritmo es el que gano por F1 macro en la tabla de 15 combinaciones del cuaderno.
# --------------------------------------------------------------------------------------
CATEGORIAS = {
    "total":       ("Hurto total (compendio)", "hurto_total",             "hurto_total_per_capita",        "Gradient Boosting"),
    "personas":    ("Hurto a personas",        "hurto_personas_total",    "hurto_personas_per_capita",     "Gradient Boosting"),
    "vehiculos":   ("Hurto de vehículos",      "hurto_vehiculos_total",   "hurto_vehiculos_per_capita",    "Gradient Boosting"),
    "residencias": ("Hurto a residencias",     "hurto_residencias_total", "hurto_residencias_per_capita",  "Gradient Boosting"),
    "comercio":    ("Hurto a comercio",        "hurto_comercio_total",    "hurto_comercio_per_capita",     "Random Forest"),
}

# Texto corto que la app muestra debajo del selector de categoria.
DESCRIPCION_CATEGORIA = {
    "total": "Suma de las cuatro categorías de hurto. Es la cifra que mejor resume "
             "'seguridad ciudadana' para comunicar, pero es **la más difícil de predecir**: "
             "al agregar, se diluye la señal de cada categoría.",
    "personas": "Atracos y hurtos a transeúntes. Es la categoría con más víctimas y la que "
                "domina la percepción de inseguridad.",
    "vehiculos": "Hurto de automotores y motocicletas. Muy concentrado en municipios grandes "
                 "y en corredores viales.",
    "residencias": "Hurto a viviendas. Junto con comercio, es **la categoría más predecible** "
                   "de las cinco con las variables disponibles.",
    "comercio": "Hurto a establecimientos comerciales. Sigue de cerca la actividad económica "
                "del municipio, no solo su tamaño.",
}

# Features de contexto, iguales en las 5 categorias (FEATURES_CONTEXTO del cuaderno).
# A cada modelo se le agrega ademas el nivel actual de SU propia categoria.
FEATURES_CONTEXTO = [
    "gasto_per_capita_log", "gasto_per_capita_lag1_log", "gasto_vigilancia_per_capita_log",
    "share_vigilancia", "tasa_competencia_promedio", "tuvo_secop", "poblacion_log",
    "homicidios_per_capita_log", "estupefacientes_per_capita_log", "capturas_per_capita_log",
]

# Columnas que se transforman con log1p antes de entrar al modelo.
COLS_LOG_CONTEXTO = [
    "gasto_per_capita", "gasto_per_capita_lag1", "gasto_vigilancia_per_capita", "poblacion",
    "homicidios_per_capita", "estupefacientes_per_capita", "capturas_per_capita",
]

# Nombres legibles: la app nunca muestra el nombre interno de una feature.
NOMBRES_LEGIBLES = {
    "gasto_per_capita_log": "Inversión en seguridad por habitante",
    "gasto_per_capita_lag1_log": "Inversión del año anterior",
    "gasto_vigilancia_per_capita_log": "Inversión específica en vigilancia armada",
    "share_vigilancia": "Porcentaje del gasto dedicado a vigilancia armada",
    "tasa_competencia_promedio": "Competencia en los procesos de contratación",
    "tuvo_secop": "Tiene contratación registrada en SECOP",
    "poblacion_log": "Tamaño del municipio",
    "homicidios_per_capita_log": "Nivel de homicidios",
    "estupefacientes_per_capita_log": "Operativos de incautación de droga",
    "capturas_per_capita_log": "Capturas realizadas por la Policía",
    "hurto_total_per_capita_log": "Nivel actual de hurto total",
    "hurto_personas_per_capita_log": "Nivel actual de hurto a personas",
    "hurto_vehiculos_per_capita_log": "Nivel actual de hurto de vehículos",
    "hurto_residencias_per_capita_log": "Nivel actual de hurto a residencias",
    "hurto_comercio_per_capita_log": "Nivel actual de hurto a comercio",
}


# ======================================================================================
#  Carga
# ======================================================================================
def _pendiente(anios, valores):
    mask = ~np.isnan(valores)
    if mask.sum() < 2:
        return np.nan
    return stats.linregress(anios[mask], valores[mask])[0]


def _signed_log1p(s):
    return np.sign(s) * np.log1p(np.abs(s))


def cargar_datos():
    """Carga el panel municipio-anio y la tabla de nombres.

    Devuelve tambien que categorias y que features de contexto estan realmente disponibles
    en el CSV: si se le pasa el dataset viejo (el de la version anterior de la app, sin
    residencias/comercio/capturas/homicidios), la app arranca igual en modo reducido en vez
    de reventar, y avisa en pantalla.
    """
    df = pd.read_csv(DATOS / "dataset_municipio_anio_final.csv", dtype={"divipola5": str})
    df["divipola5"] = df["divipola5"].str.zfill(5)
    df = df[df["anio"] >= ANIO_INICIO].copy()

    # --- Derivaciones defensivas: el cuaderno ya las hace, pero un CSV intermedio puede no
    #     traer alguna. Reconstruirlas aqui evita depender del orden en que se regeneren.
    totales_hurto = ["hurto_personas_total", "hurto_vehiculos_total",
                     "hurto_residencias_total", "hurto_comercio_total"]
    if "hurto_total" not in df.columns and all(c in df.columns for c in totales_hurto):
        df["hurto_total"] = df[totales_hurto].fillna(0).sum(axis=1)

    for _, col_total, col_pc, _ in CATEGORIAS.values():
        if col_pc not in df.columns and col_total in df.columns and "poblacion" in df.columns:
            df[col_pc] = df[col_total] / df["poblacion"] * 10000
    for base in ["capturas", "homicidios"]:
        col_pc, col_total = f"{base}_per_capita", f"{base}_total"
        if col_pc not in df.columns and col_total in df.columns:
            df[col_pc] = df[col_total] / df["poblacion"] * 10000

    categorias = {k: v for k, v in CATEGORIAS.items() if v[2] in df.columns}
    features_contexto = [f for f in FEATURES_CONTEXTO
                         if f.replace("_log", "") in df.columns
                         or f in ("share_vigilancia", "tasa_competencia_promedio", "tuvo_secop")
                         or f in ("gasto_per_capita_lag1_log", "poblacion_log")]
    # `gasto_per_capita_lag1` se construye mas adelante; `poblacion_log` sale de `poblacion`.
    if "poblacion" not in df.columns:
        features_contexto = [f for f in features_contexto if f != "poblacion_log"]

    nombres = pd.read_csv(DATOS / "municipios.csv", dtype={"divipola5": str})
    nombres["divipola5"] = nombres["divipola5"].str.zfill(5)

    faltantes = [v[0] for k, v in CATEGORIAS.items() if k not in categorias]
    return df, nombres, categorias, features_contexto, faltantes


# ======================================================================================
#  Perfiles y clustering (seccion II.5 y II.7), ahora por categoria
# ======================================================================================
def construir_perfiles(dm, nombres, cat_key):
    """Un registro por municipio para LA categoria elegida: nivel, tendencia, cuadrante y cluster."""
    _, _, col_pc, _ = CATEGORIAS[cat_key]

    # Promedio de TODAS las categorias disponibles, no solo la elegida: la app muestra un
    # desglose de las cuatro para que el funcionario vea de un vistazo si el problema es
    # general o de una sola categoria. No sale del cuaderno; es utilidad de interfaz.
    cols_desglose = {k: v[2] for k, v in CATEGORIAS.items() if v[2] in dm.columns}

    filas = []
    for div, g in dm.groupby("divipola5"):
        g = g.sort_values("anio")
        a = g["anio"].values.astype(float)
        fila = {
            "divipola5": div,
            "gasto_pc_prom": g["gasto_per_capita"].mean(),
            "gasto_pc_tend": _pendiente(a, g["gasto_per_capita"].values),
            "cat_pc_prom": g[col_pc].mean(),
            "cat_pc_tend": _pendiente(a, g[col_pc].values),
            "tasa_competencia_prom": g["tasa_competencia_promedio"].mean(),
            "poblacion_prom": g["poblacion"].mean(),
            "anios_con_secop": int(g["tuvo_secop"].sum()),
            "anios_totales": len(g),
        }
        for k, c in cols_desglose.items():
            fila[f"prom_{k}"] = g[c].mean()
        filas.append(fila)
    p = pd.DataFrame(filas)
    p["tasa_competencia_prom"] = p["tasa_competencia_prom"].fillna(p["tasa_competencia_prom"].median())
    p = p.dropna(subset=["gasto_pc_prom", "cat_pc_prom", "poblacion_prom"]).reset_index(drop=True)

    # --- Cuadrantes (linea base descriptiva, seccion II.5) ---
    # CAMBIO respecto al cuaderno: el umbral de gasto se calcula entre los municipios CON
    # dato, no sobre todos. Con el dataset nuevo, 732 de 1.110 municipios no tienen ningun
    # proceso SECOP, asi que la mediana de gasto de TODOS es exactamente 0 y "gasto >=
    # mediana" se vuelve verdadero para todo el mundo: el cuaderno reporta 555/555 repartidos
    # solo por hurto, y los cuadrantes "bajo gasto" quedan vacios. Ese 555/555 es un artefacto
    # de cobertura, no un hallazgo, y presentarselo a FONSECON como cuadrante seria enganoso.
    con_dato = p["anios_con_secop"] > 0
    med_gasto = p.loc[con_dato, "gasto_pc_prom"].median()
    med_cat = p["cat_pc_prom"].median()
    p["cuadrante"] = np.select(
        [
            ~con_dato,
            (p["gasto_pc_prom"] < med_gasto) & (p["cat_pc_prom"] >= med_cat),
            (p["gasto_pc_prom"] >= med_gasto) & (p["cat_pc_prom"] < med_cat),
            (p["gasto_pc_prom"] >= med_gasto) & (p["cat_pc_prom"] >= med_cat),
        ],
        ["sin dato de gasto", "bajo gasto / alto hurto", "alto gasto / bajo hurto",
         "alto gasto / alto hurto"],
        default="bajo gasto / bajo hurto")

    # --- Clustering (seccion II.7) ---
    p["gasto_pc_prom_log"] = np.log1p(p["gasto_pc_prom"])
    p["gasto_pc_tend_slog"] = _signed_log1p(p["gasto_pc_tend"])
    p["cat_pc_prom_log"] = np.log1p(p["cat_pc_prom"])
    p["cat_pc_tend_slog"] = _signed_log1p(p["cat_pc_tend"])
    p["poblacion_prom_log"] = np.log1p(p["poblacion_prom"])

    features_cluster = ["gasto_pc_prom_log", "gasto_pc_tend_slog", "cat_pc_prom_log",
                        "cat_pc_tend_slog", "tasa_competencia_prom", "poblacion_prom_log"]
    Xc = p[features_cluster].fillna(p[features_cluster].median())
    p["cluster"] = KMeans(K_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit_predict(
        StandardScaler().fit_transform(Xc))

    p["etiquetas_cluster"] = p["cluster"].map(_etiquetar_clusters(p))
    p = p.merge(nombres, on="divipola5", how="left")
    # La tabla canonica sale de los datos de la Policia: le faltan municipios sin casos
    # registrados. Se muestran por su codigo en vez de dejar un "nan" en el selector.
    p["municipio"] = p["municipio"].fillna("Municipio " + p["divipola5"])
    p["departamento"] = p["departamento"].fillna("(departamento no identificado)")
    return p


def _etiquetar_clusters(p):
    """Etiqueta cada cluster por la posicion real de sus municipios (seccion II.7).

    El umbral de "gasto alto" se calcula solo entre municipios CON dato de gasto: usar la
    mediana de todos haria que "gasto alto" significara apenas "gasto algo".
    """
    con_dato = p["anios_con_secop"] > 0
    u_gasto = p.loc[con_dato, "gasto_pc_prom"].median()
    u_cat = p["cat_pc_prom"].median()
    diag = p.assign(
        ga=p["gasto_pc_prom"] >= u_gasto,
        ha=p["cat_pc_prom"] >= u_cat,
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


# ======================================================================================
#  Modelado supervisado (seccion II.6), por categoria
# ======================================================================================
def preparar_modelado(dm, cat_key, features_contexto):
    """Target de cambio t -> t+1 de LA categoria, con terciles calculados solo en train."""
    _, _, col_pc, _ = CATEGORIAS[cat_key]

    d = dm.sort_values(["divipola5", "anio"]).copy()
    d["target_siguiente"] = d.groupby("divipola5")[col_pc].shift(-1)
    d["anio_siguiente"] = d.groupby("divipola5")["anio"].shift(-1)
    d["cambio"] = d["target_siguiente"] - d[col_pc]
    d["gasto_per_capita_lag1"] = d.groupby("divipola5")["gasto_per_capita"].shift(1).fillna(0)

    entrenable = d[(d["anio_siguiente"] == d["anio"] + 1)
                   & d[col_pc].notna() & d["target_siguiente"].notna()].copy()

    # Cortes de clase calculados SOLO con anios de entrenamiento: usarlos sobre todo el panel
    # seria filtrar informacion del futuro hacia la definicion misma del target.
    es_train = entrenable["anio"] <= ANIO_CORTE_TRAIN
    cortes = entrenable.loc[es_train, "cambio"].quantile([1 / 3, 2 / 3]).values
    entrenable["clase"] = pd.cut(entrenable["cambio"],
                                 [-np.inf, cortes[0], cortes[1], np.inf],
                                 labels=CLASES)

    # Las filas del ultimo anio no tienen target, pero si sirven para predecir hacia adelante.
    ultimo = d.groupby("divipola5").tail(1).copy()
    ultimo["clase"] = None

    for frame in (entrenable, ultimo):
        for col in ["tasa_competencia_promedio", "share_vigilancia"]:
            frame[col] = frame[col].fillna(entrenable.loc[es_train, col].median())
        for c in COLS_LOG_CONTEXTO + [col_pc]:
            if c in frame.columns:
                frame[c + "_log"] = np.log1p(frame[c].clip(lower=0))

    # Regla de exclusion de II.6: solo el nivel de la PROPIA categoria entra como control.
    features = list(features_contexto) + [col_pc + "_log"]
    return entrenable, ultimo, features


def entrenar(entrenable, features, cat_key):
    """Entrena el algoritmo ganador de esa categoria segun la tabla de 15 combinaciones."""
    _, _, _, algoritmo = CATEGORIAS[cat_key]

    train = entrenable[entrenable["anio"] <= ANIO_CORTE_TRAIN]
    test = entrenable[entrenable["anio"] > ANIO_CORTE_TRAIN]

    if algoritmo == "Random Forest":
        modelo = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                        random_state=RANDOM_STATE, n_jobs=-1)
    else:
        modelo = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                            random_state=RANDOM_STATE)
    modelo.fit(train[features], train["clase"].astype(str))

    pred_test = modelo.predict(test[features])
    real_test = test["clase"].astype(str).values
    marcados = pred_test == "empeoro"

    # F1 macro sin importar sklearn.metrics: es el promedio simple del F1 de las 3 clases.
    f1s = []
    for cl in CLASES:
        tp = int(((pred_test == cl) & (real_test == cl)).sum())
        fp = int(((pred_test == cl) & (real_test != cl)).sum())
        fn = int(((pred_test != cl) & (real_test == cl)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))

    metricas = {
        "algoritmo": algoritmo,
        "precision_alerta": float((real_test[marcados] == "empeoro").mean()) if marcados.any() else 0.0,
        "n_marcados": int(marcados.sum()),
        "accuracy": float((pred_test == real_test).mean()),
        "f1_macro": float(np.mean(f1s)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "anios_test": f"{ANIO_CORTE_TRAIN + 1}-{int(entrenable['anio'].max())}",
    }
    return modelo, metricas


# ======================================================================================
#  Explicabilidad (SHAP) — seccion II.6 del cuaderno
# ======================================================================================
def crear_explicador(modelo, entrenable, features):
    """Devuelve una funcion `X -> (contribuciones hacia 'empeoro', titulo)`.

    `shap.TreeExplainer` NO soporta `GradientBoostingClassifier` multiclase (lanza
    InvalidModelError), y Gradient Boosting es el algoritmo ganador en 4 de las 5
    categorias. Antes de aceptar la degradacion a "importancia general" —que es global y
    no explica el caso concreto que el funcionario esta mirando— se intenta el
    `Permutation` explainer, que es agnostico al modelo. Cuesta unos segundos la primera
    vez (compilacion interna de shap) y es instantaneo despues, por eso aqui se hace una
    llamada de calentamiento: que el costo lo pague el spinner de carga y no el primer
    municipio que el usuario consulte.
    """
    idx_empeoro = list(modelo.classes_).index("empeoro")
    titulo_local = "Cuánto empuja cada factor hacia un empeoramiento"

    def _fallback(X):
        return pd.Series(modelo.feature_importances_, index=features), \
            "Factores que más pesan en el modelo (importancia general)"

    try:
        import shap
    except Exception:
        return _fallback

    muestra = entrenable[features].head(1)

    # 1) TreeExplainer: exacto y rapido. Sirve para Random Forest (categoria Comercio).
    try:
        explainer = shap.TreeExplainer(modelo)

        def _tree(X):
            valores = explainer.shap_values(X)
            if isinstance(valores, list):              # shap antiguo: una matriz por clase
                v = np.asarray(valores[idx_empeoro])[0]
            else:
                arr = np.asarray(valores)
                v = arr[0][:, idx_empeoro] if arr.ndim == 3 else arr[0]
            return pd.Series(v, index=features), titulo_local

        _tree(muestra)
        return _tree
    except Exception:
        pass

    # 2) Permutation: agnostico al modelo, para Gradient Boosting multiclase.
    try:
        fondo = shap.sample(entrenable[features], 40, random_state=RANDOM_STATE)
        explainer = shap.explainers.Permutation(modelo.predict_proba, fondo,
                                                max_evals=2 * len(features) + 1)

        def _perm(X):
            v = np.asarray(explainer(X).values)[0][:, idx_empeoro]
            return pd.Series(v, index=features), titulo_local

        _perm(muestra)          # calentamiento
        return _perm
    except Exception:
        return _fallback
