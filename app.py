"""
Consulta de Seguridad Municipal — prototipo para FONSECON.

Fase 3 del proyecto "Inversión Pública en Seguridad vs. Criminalidad Municipal en Colombia"
(Técnicas de Aprendizaje de Máquina, Pontificia Universidad Javeriana).

ACTUALIZADA a la Parte 4 rediseñada del cuaderno: el proyecto ya no analiza una sola variable
de resultado (hurto a personas) sino el **compendio de hurto y sus cuatro categorías**. La app
recoge ese cambio con un selector de categoría: el funcionario ya no pregunta "¿cómo va la
seguridad aquí?" sino "¿qué tipo de hurto está empeorando aquí, y qué tan confiable es esa
señal en cada caso?".

Está diseñada para un funcionario que necesita decidir dónde mirar, no para un científico de
datos: no muestra métricas sin traducir, dice explícitamente qué tan confiable es cada
resultado, y cuando no hay información lo dice en vez de mostrar un cero.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import modelo as M

st.set_page_config(page_title="Consulta de Seguridad Municipal",
                   page_icon="🛡️", layout="wide")

COLOR_GASTO, COLOR_HURTO = "#2a6f97", "#d62828"


# --------------------------------------------------------------------------- carga
@st.cache_data(show_spinner="Cargando el panel municipio-año...")
def _datos():
    return M.cargar_datos()


@st.cache_data(show_spinner="Construyendo perfiles de municipio...")
def _perfiles(_dm, _nombres, cat_key):
    return M.construir_perfiles(_dm, _nombres, cat_key)


@st.cache_resource(show_spinner="Entrenando el modelo de esta categoría...")
def _modelo(_dm, cat_key, _features_contexto):
    entrenable, ultimo, features = M.preparar_modelado(_dm, cat_key, _features_contexto)
    clf, metricas = M.entrenar(entrenable, features, cat_key)
    explicar = M.crear_explicador(clf, entrenable, features)
    return clf, metricas, features, ultimo, explicar


dm, nombres, CATS, FEATURES_CONTEXTO, FALTANTES = _datos()


# --------------------------------------------------------------------------- cabecera
st.title("🛡️ Consulta de Seguridad Municipal")
st.markdown(
    "**Para qué sirve:** ayuda a decidir *en qué municipios y sobre qué tipo de hurto mirar "
    "primero* al asignar o auditar recursos de seguridad, cruzando la contratación pública de "
    "vigilancia (SECOP II) con las cifras de criminalidad de la Policía Nacional entre 2016 y "
    "2025."
)

if FALTANTES:
    st.error(
        "**Datos desactualizados.** El archivo `datos/dataset_municipio_anio_final.csv` que está "
        "usando la app es la versión anterior del panel y no trae "
        f"{', '.join(FALTANTES)}. La app funciona, pero solo con las categorías que sí encontró. "
        "Para habilitar las cinco, copia el CSV que genera la sección II.4 del cuaderno "
        "(`data/processed/dataset_municipio_anio_final.csv`) dentro de la carpeta `datos/`."
    )

with st.expander("Antes de usarla, lee esto (2 minutos)"):
    st.markdown("""
**Qué hace exactamente.** Eliges un **tipo de hurto** y un **municipio**, y la herramienta
responde tres preguntas sobre esa combinación: ¿cómo se compara su inversión en seguridad con su
nivel de ese hurto?, ¿a qué otros municipios se parece?, y ¿ese hurto tiende a empeorar el
próximo año?

**Por qué hay cinco categorías y no una.** El proyecto resuelve el mismo problema cinco veces en
paralelo —Personas, Vehículos, Residencias, Comercio y el compendio Total— porque **no son
igual de predecibles**. Residencias y Comercio se anticipan bastante mejor que el agregado; el
compendio Total, que es la cifra más cómoda para comunicar, es justamente **la peor** para
predecir. La app muestra la confiabilidad real de cada categoría en vez de esconder esa
diferencia detrás de un solo número.

**Qué NO hace, y es importante.** No dice cuánto invertir ni en qué. Y sobre todo: **no demuestra
que invertir reduzca el hurto**. En los datos colombianos los municipios que más gastan son los
que más hurto tienen — no porque la vigilancia empeore la seguridad, sino porque **se invierte
donde ya hay problema**. Cualquier lectura causal de estos números es incorrecta.

**El modelo de una categoría no usa las demás como pista.** Si estás mirando Hurto de Vehículos,
el modelo no sabe cuánto hurto a personas hay en ese municipio. Predecir hurto con hurto daría
métricas bonitas y ninguna utilidad para FONSECON, que no controla el hurto: controla el
presupuesto.
""")

st.divider()


# --------------------------------------------------------------------------- selectores
col_cat, col_sel, col_info = st.columns([2, 2, 3])

with col_cat:
    etiquetas_cat = {v[0]: k for k, v in CATS.items()}
    etiqueta_elegida = st.selectbox("Tipo de hurto", list(etiquetas_cat.keys()))
    cat_key = etiquetas_cat[etiqueta_elegida]

etiqueta_cat, _, col_pc, algoritmo_cat = M.CATEGORIAS[cat_key]

perfiles = _perfiles(dm, nombres, cat_key)
clf, metricas, FEATURES, ultimo, explicar = _modelo(dm, cat_key, FEATURES_CONTEXTO)

perfiles = perfiles.sort_values(["departamento", "municipio"])
perfiles["etiqueta"] = perfiles["municipio"] + " — " + perfiles["departamento"]

with col_sel:
    eleccion = st.selectbox("Municipio", perfiles["etiqueta"].tolist(),
                            index=int(np.argmax(perfiles["poblacion_prom"].values)))

p = perfiles[perfiles["etiqueta"] == eleccion].iloc[0]
div = p["divipola5"]
hist = dm[dm["divipola5"] == div].sort_values("anio")
sin_datos_gasto = p["anios_con_secop"] == 0

with col_info:
    st.markdown(f"### {p['municipio']}, {p['departamento']}")
    st.caption(f"Código DIVIPOLA {div}  ·  {p['poblacion_prom']:,.0f} habitantes (promedio 2016-2025)")

st.caption(M.DESCRIPCION_CATEGORIA[cat_key])

if sin_datos_gasto:
    st.warning(
        "**Este municipio no tiene ningún contrato de seguridad registrado en SECOP entre 2016 y "
        "2025.** No sabemos cuánto invierte. Todo lo que aparezca abajo como inversión es un "
        "vacío de información, no un cero real — y lo que este municipio necesita primero es una "
        "**revisión de por qué no está reportando**, no necesariamente más presupuesto."
    )
elif p["anios_con_secop"] < p["anios_totales"] / 2:
    st.info(
        f"Cobertura parcial: solo **{p['anios_con_secop']} de {p['anios_totales']} años** tienen "
        "contratación registrada. Los años en blanco son falta de dato, no ausencia de inversión."
    )


# --------------------------------------------------------------------------- indicadores
st.subheader("Situación del municipio")

# El cuarto indicador (desglose) solo aparece si el panel trae más de una categoría.
CATS_DESGLOSE = [k for k in CATS if f"prom_{k}" in perfiles.columns]
c1, c2, c3, *resto = st.columns(4 if len(CATS_DESGLOSE) > 1 else 3)
c4 = resto[0] if resto else None

c1.metric(etiqueta_cat, f"{p['cat_pc_prom']:.1f}",
          help="Casos por cada 10.000 habitantes al año, promedio 2016-2025.")
c1.caption(f"Mediana nacional: {perfiles['cat_pc_prom'].median():.1f}")

if sin_datos_gasto:
    c2.metric("Inversión en seguridad", "Sin dato")
    c2.caption("Ningún proceso registrado en SECOP")
else:
    c2.metric("Inversión en seguridad", f"${p['gasto_pc_prom']:,.0f}",
              help="Pesos por habitante al año, promedio de los años con contratación registrada.")
    c2.caption(f"Mediana de los municipios con dato: "
               f"${perfiles.loc[perfiles['anios_con_secop'] > 0, 'gasto_pc_prom'].median():,.0f}")

tendencia = p["cat_pc_tend"]
c3.metric("Tendencia", "En aumento" if tendencia > 0.1 else
          ("A la baja" if tendencia < -0.1 else "Estable"),
          delta=f"{tendencia:+.2f} por año", delta_color="inverse",
          help=f"Pendiente de la recta ajustada a {etiqueta_cat.lower()} per cápita, 2016-2025.")

if c4 is not None:
    ABREV = {"personas": "Pers.", "vehiculos": "Veh.", "residencias": "Res.",
             "comercio": "Com.", "total": "Total"}
    if "total" in CATS_DESGLOSE:
        c4.metric("Hurto total", f"{p['prom_total']:.1f}",
                  help="Compendio de las cuatro categorías, por 10.000 habitantes.")
    else:
        c4.metric("Desglose", "por categoría")
    c4.caption(" · ".join(f"{ABREV[k]} {p[f'prom_{k}']:.1f}"
                          for k in CATS_DESGLOSE if k != "total"))


# --------------------------------------------------------------------------- gráfica
fig, ax1 = plt.subplots(figsize=(9, 3.6))
ax1.plot(hist["anio"], hist[col_pc], marker="o",
         color=COLOR_HURTO, linewidth=2, label=etiqueta_cat)
ax1.set_ylabel(f"{etiqueta_cat} (x10.000 hab.)", color=COLOR_HURTO)
ax1.tick_params(axis="y", labelcolor=COLOR_HURTO)
ax1.set_xlabel("Año")

ax2 = ax1.twinx()
con_gasto = hist[hist["tuvo_secop"] == 1]
ax2.bar(con_gasto["anio"], con_gasto["gasto_per_capita"], alpha=0.35,
        color=COLOR_GASTO, label="Inversión registrada")
ax2.set_ylabel("Inversión ($/hab.)", color=COLOR_GASTO)
ax2.tick_params(axis="y", labelcolor=COLOR_GASTO)

for anio in hist.loc[hist["tuvo_secop"] == 0, "anio"]:
    ax1.axvspan(anio - 0.5, anio + 0.5, color="lightgray", alpha=0.3, zorder=0)

ax1.set_title(f"Inversión registrada (barras) y {etiqueta_cat.lower()} (línea).\n"
              "Las franjas grises son años sin datos de contratación.", fontsize=9)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------------------------------- perfil
st.subheader("¿A qué otros municipios se parece?")
col_a, col_b = st.columns(2)
col_a.markdown(f"**Comparación simple:** `{p['cuadrante']}`")
col_a.caption("Compara al municipio contra la mediana nacional de gasto (entre los que sí "
              "reportan) y la mediana de esta categoría de hurto.")
col_b.markdown(f"**Grupo por perfil completo:** `{p['etiquetas_cluster']}`")
col_b.caption("Agrupa municipios con historia parecida de gasto, de esta categoría de hurto y "
              "de su tendencia.")

similares = perfiles[(perfiles["cluster"] == p["cluster"]) & (perfiles["divipola5"] != div)]
similares = similares.reindex(
    (similares["poblacion_prom"] - p["poblacion_prom"]).abs().sort_values().index).head(5)
st.dataframe(
    similares[["municipio", "departamento", "gasto_pc_prom", "cat_pc_prom", "anios_con_secop"]]
    .rename(columns={"municipio": "Municipio", "departamento": "Departamento",
                     "gasto_pc_prom": "Inversión $/hab.", "cat_pc_prom": f"{etiqueta_cat} x10.000",
                     "anios_con_secop": "Años con dato"}),
    hide_index=True, width="stretch")


# --------------------------------------------------------------------------- predicción
st.subheader("Previsión para el próximo año")
fila = ultimo[ultimo["divipola5"] == div]

# Si al último año del municipio le falta alguna feature, el modelo aún devolvería una
# predicción (los árboles de sklearn no fallan con NaN), pero sería basura presentada con
# el mismo aspecto que una buena. Mejor decir que no se puede.
if fila.empty or fila[FEATURES].isna().any(axis=1).iloc[0]:
    st.info("No hay datos suficientes para hacer una previsión de este municipio en esta "
            "categoría.")
else:
    X = fila[FEATURES]
    prediccion = clf.predict(X)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    anio_base = int(fila["anio"].iloc[0])

    texto = {"empeoro": (f"🔴 {etiqueta_cat} podría **aumentar**", "error"),
             "se_mantuvo": (f"🟡 {etiqueta_cat} se **mantendría** estable", "warning"),
             "mejoro": (f"🟢 {etiqueta_cat} podría **bajar**", "success")}[prediccion]
    getattr(st, texto[1])(
        f"{texto[0]} en {anio_base + 1}, respecto a {anio_base}.  \n"
        f"Confianza del modelo en este resultado: **{proba[prediccion] * 100:.0f}%** "
        f"(un resultado al azar daría 33%)."
    )

    st.markdown("**Por qué el modelo dice eso**")
    contrib, titulo = explicar(X)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)[::-1]
    fig2, ax = plt.subplots(figsize=(8, 2.8))
    ax.barh([M.NOMBRES_LEGIBLES.get(i, i) for i in top.index], top.values,
            color=[COLOR_HURTO if v > 0 else "#2a9d8f" for v in top.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(titulo + "\n(rojo: empuja hacia empeorar · verde: empuja hacia mejorar)",
                 fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        f"⚠️ En {etiqueta_cat.lower()}, esta previsión acierta aproximadamente "
        f"**{metricas['precision_alerta'] * 100:.0f} de cada 100 veces** que marca un municipio "
        f"en rojo (modelo: {metricas['algoritmo']}; prueba sobre {metricas['anios_test']}, "
        f"{metricas['n_test']:,} municipio-años). Úsala para decidir a quién llamar primero, no "
        "para asignar presupuesto."
    )


# --------------------------------------------------------------------------- comparación
st.subheader("¿En qué categoría es más confiable la alerta?")
st.caption("Entrena los modelos de las demás categorías y compara su desempeño sobre los mismos "
           "años de prueba. Tarda unos segundos la primera vez.")

if st.checkbox("Comparar todas las categorías"):
    filas = []
    for k, (label, _, _, algo) in M.CATEGORIAS.items():
        if k not in CATS:
            continue
        _, met_k, _, _, _ = _modelo(dm, k, FEATURES_CONTEXTO)
        filas.append({
            "Tipo de hurto": label,
            "Algoritmo": met_k["algoritmo"],
            "Aciertos globales": f"{met_k['accuracy'] * 100:.0f}%",
            "Equilibrio entre clases (F1)": f"{met_k['f1_macro']:.2f}",
            "Aciertos de la alerta roja": f"{met_k['precision_alerta'] * 100:.0f}%",
        })
    tabla = pd.DataFrame(filas)
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.caption(
        "Un modelo que adivinara al azar acertaría 33% de las veces. **Residencias y comercio "
        "son las categorías más predecibles; el compendio total es la menos predecible** — "
        "agregarlo todo en una sola cifra cuesta poder predictivo, no es gratis."
    )


# --------------------------------------------------------------------------- acciones
st.subheader("Qué hacer con este resultado")

if sin_datos_gasto:
    st.markdown("""
**Acción recomendada: auditoría de reporte, no asignación de recursos.**

1. Verificar con la alcaldía si realmente no contrata servicios de seguridad, o si los contrata
   por fuera de SECOP II o sin el código UNSPSC correspondiente.
2. Si contrata y no reporta, el problema es de capacidad administrativa: acompañamiento en el
   uso de la plataforma antes que cofinanciación.
3. Solo después de cerrar ese vacío tiene sentido evaluarlo para asignación de recursos.

*Este caso es más común de lo que parece: aplica a 2 de cada 3 municipios del país, y el patrón
se repite idéntico en las cinco categorías de hurto — no es una particularidad de ninguna.*
""")
elif p["cuadrante"] == "bajo gasto / alto hurto":
    st.markdown(f"""
**Acción recomendada: candidato prioritario a cofinanciación.**

1. Es de los pocos municipios con **inversión baja verificada** y {etiqueta_cat.lower()} alto —
   el perfil que FONSECON busca, y que sí tiene datos que lo respalden.
2. Revisar si la baja inversión es por falta de recursos propios o por baja capacidad de
   formulación de proyectos: la respuesta cambia entre girar plata y dar asistencia técnica.
3. Antes de decidir, mirar las otras categorías de hurto en el comparador de arriba: un
   municipio puede estar bien en el agregado y mal en una categoría concreta, y el tipo de
   intervención no es el mismo para hurto a residencias que para hurto a personas.
""")
elif p["cuadrante"] == "alto gasto / alto hurto":
    st.markdown(f"""
**Acción recomendada: revisar antes de aumentar el presupuesto.**

1. Invertir más aquí probablemente no cambie el resultado por sí solo: este es el perfil típico
   de **gasto reactivo**, donde la inversión responde al problema en vez de anticiparlo.
2. Revisar la **composición** del gasto (cuánto es vigilancia armada y cuánto otros rubros) y
   los tiempos de adjudicación, antes que el monto total.
3. Comprobar si el problema es de todas las categorías o solo de {etiqueta_cat.lower()}: si es
   de una sola, la respuesta es focalizada, no un aumento general del presupuesto.
""")
elif p["cuadrante"] == "alto gasto / bajo hurto":
    st.markdown("""
**Acción recomendada: revisar eficiencia, con cautela.**

1. Gasta por encima de la mediana y tiene poco hurto de este tipo. Antes de leerlo como éxito,
   verificar que no sea un municipio pequeño donde **un solo contrato** distorsiona la cifra por
   habitante.
2. Si la inversión es sostenida en el tiempo, vale la pena documentar qué está haciendo: puede
   ser un caso replicable.
3. Contrastar con las demás categorías antes de declararlo buen ejemplo: es frecuente estar bien
   en una y mal en otra.
""")
else:
    st.markdown("""
**Acción recomendada: seguimiento de rutina.**

1. Inversión y hurto de este tipo por debajo de la mediana nacional. No es prioritario para
   asignación ni para auditoría en este momento.
2. Vigilar la tendencia: si empieza a subir de forma sostenida, reevaluar.
""")

st.divider()
st.caption(
    "Prototipo académico — Técnicas de Aprendizaje de Máquina, Pontificia Universidad Javeriana. "
    "Fuentes: SECOP II y Policía Nacional (datos.gov.co), proyecciones de población del DANE. "
    "Las cifras de hurto corresponden a **denuncias registradas**, no al delito real: se estima "
    "que una fracción importante de los hurtos no se denuncia. No usar para decisiones "
    "presupuestales sin validación adicional."
)
