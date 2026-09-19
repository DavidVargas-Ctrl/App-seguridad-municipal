"""
Consulta de Seguridad Municipal — prototipo para FONSECON.

Fase 3 del proyecto "Inversión Pública en Seguridad vs. Criminalidad Municipal en Colombia"
(Técnicas de Aprendizaje de Máquina, Pontificia Universidad Javeriana).

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
@st.cache_data(show_spinner="Cargando datos de los 1.110 municipios...")
def _datos():
    dm, nombres = M.cargar_datos()
    perfiles = M.construir_perfiles(dm, nombres)
    entrenable, ultimo = M.preparar_modelado(dm)
    return dm, nombres, perfiles, entrenable, ultimo


@st.cache_resource(show_spinner="Entrenando el modelo...")
def _modelo(_entrenable):
    return M.entrenar(_entrenable)


dm, nombres, perfiles, entrenable, ultimo = _datos()
clf, metricas = _modelo(entrenable)


# --------------------------------------------------------------------------- cabecera
st.title("🛡️ Consulta de Seguridad Municipal")
st.markdown(
    "**Para qué sirve:** ayuda a decidir *en qué municipios mirar primero* al asignar o auditar "
    "recursos de seguridad, cruzando la contratación pública de vigilancia (SECOP II) con las "
    "cifras de criminalidad de la Policía Nacional entre 2016 y 2025."
)

with st.expander("Antes de usarla, lee esto (2 minutos)"):
    st.markdown(f"""
**Qué hace exactamente.** Para el municipio que elijas, la herramienta responde tres preguntas:
¿cómo se compara su inversión en seguridad con su nivel de hurto?, ¿a qué otros municipios se
parece?, y ¿el hurto tiende a empeorar el próximo año?

**Qué NO hace, y es importante.** No dice cuánto invertir ni en qué. Y sobre todo: **no demuestra
que invertir reduzca el hurto**. En los datos colombianos, los municipios que más gastan son los
que más hurto tienen — no porque la vigilancia empeore la seguridad, sino porque **se invierte
donde ya hay problema**. Cualquier lectura causal de estos números es incorrecta.

**Qué tan confiable es la alerta de riesgo.** Poco, y por eso lo decimos aquí arriba. De cada
100 municipios que el modelo marca como "va a empeorar", aciertan alrededor de
**{metricas['precision_alerta'] * 100:.0f}**. Sirve como **filtro** para reducir 1.110 municipios
a una lista corta que alguien revise, nunca como decisión automática.

**La limitación más grave.** {(perfiles['anios_con_secop'] == 0).sum()} de {len(perfiles)}
municipios no tienen **ningún** contrato de seguridad registrado en SECOP en todo el período.
Para ellos "inversión 0" significa *no tenemos el dato*, no *no invierten*. La herramienta te
avisa cuando estás viendo uno de esos casos.
""")

st.divider()


# --------------------------------------------------------------------------- selector
perfiles = perfiles.sort_values(["departamento", "municipio"])
perfiles["etiqueta"] = perfiles["municipio"] + " — " + perfiles["departamento"]

col_sel, col_info = st.columns([2, 3])
with col_sel:
    eleccion = st.selectbox("Selecciona un municipio", perfiles["etiqueta"].tolist(),
                            index=int(np.argmax(perfiles["poblacion_prom"].values)))
p = perfiles[perfiles["etiqueta"] == eleccion].iloc[0]
div = p["divipola5"]
hist = dm[dm["divipola5"] == div].sort_values("anio")
sin_datos_gasto = p["anios_con_secop"] == 0

with col_info:
    st.markdown(f"### {p['municipio']}, {p['departamento']}")
    st.caption(f"Código DIVIPOLA {div}  ·  {p['poblacion_prom']:,.0f} habitantes (promedio 2016-2025)")

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
c1, c2, c3 = st.columns(3)
c1.metric("Hurto a personas", f"{p['hurto_pc_prom']:.1f}",
          help="Víctimas por cada 10.000 habitantes al año, promedio 2016-2025.")
c1.caption(f"Mediana nacional: {perfiles['hurto_pc_prom'].median():.1f}")

if sin_datos_gasto:
    c2.metric("Inversión en seguridad", "Sin dato")
    c2.caption("Ningún proceso registrado en SECOP")
else:
    c2.metric("Inversión en seguridad", f"${p['gasto_pc_prom']:,.0f}",
              help="Pesos por habitante al año, promedio de los años con contratación registrada.")
    c2.caption(f"Mediana de los municipios con dato: "
               f"${perfiles.loc[perfiles['anios_con_secop'] > 0, 'gasto_pc_prom'].median():,.0f}")

tendencia = p["hurto_pc_tend"]
c3.metric("Tendencia del hurto", "En aumento" if tendencia > 0.1 else
          ("A la baja" if tendencia < -0.1 else "Estable"),
          delta=f"{tendencia:+.2f} por año", delta_color="inverse",
          help="Pendiente de la recta ajustada al hurto per cápita entre 2016 y 2025.")


# --------------------------------------------------------------------------- gráfica
fig, ax1 = plt.subplots(figsize=(9, 3.6))
ax1.plot(hist["anio"], hist["hurto_personas_per_capita"], marker="o",
         color=COLOR_HURTO, linewidth=2, label="Hurto a personas")
ax1.set_ylabel("Hurto por 10.000 hab.", color=COLOR_HURTO)
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

ax1.set_title("Inversión registrada (barras) y hurto a personas (línea).\n"
              "Las franjas grises son años sin datos de contratación.", fontsize=9)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------------------------------- perfil
st.subheader("¿A qué otros municipios se parece?")
col_a, col_b = st.columns(2)
col_a.markdown(f"**Comparación simple:** `{p['cuadrante']}`")
col_a.caption("Compara al municipio contra la mediana nacional de gasto y de hurto.")
col_b.markdown(f"**Grupo por perfil completo:** `{p['etiquetas_cluster']}`")
col_b.caption("Agrupa municipios con historia parecida de gasto, hurto y tendencia.")

similares = perfiles[(perfiles["cluster"] == p["cluster"]) & (perfiles["divipola5"] != div)]
similares = similares.reindex(
    (similares["poblacion_prom"] - p["poblacion_prom"]).abs().sort_values().index).head(5)
st.dataframe(
    similares[["municipio", "departamento", "gasto_pc_prom", "hurto_pc_prom", "anios_con_secop"]]
    .rename(columns={"municipio": "Municipio", "departamento": "Departamento",
                     "gasto_pc_prom": "Inversión $/hab.", "hurto_pc_prom": "Hurto x10.000",
                     "anios_con_secop": "Años con dato"}),
    hide_index=True, width="stretch")


# --------------------------------------------------------------------------- predicción
st.subheader("Previsión para el próximo año")
fila = ultimo[ultimo["divipola5"] == div]
if fila.empty:
    st.info("No hay datos suficientes para hacer una previsión de este municipio.")
    prediccion = None
else:
    X = fila[M.FEATURES]
    prediccion = clf.predict(X)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    anio_base = int(fila["anio"].iloc[0])

    texto = {"empeoro": ("🔴 El hurto podría **aumentar**", "error"),
             "se_mantuvo": ("🟡 El hurto se **mantendría** estable", "warning"),
             "mejoro": ("🟢 El hurto podría **bajar**", "success")}[prediccion]
    getattr(st, texto[1])(
        f"{texto[0]} en {anio_base + 1}, respecto a {anio_base}.  \n"
        f"Confianza del modelo en este resultado: **{proba[prediccion] * 100:.0f}%** "
        f"(un resultado al azar daría 33%)."
    )

    # --- Explicabilidad: por que dijo eso ---
    st.markdown("**Por qué el modelo dice eso**")
    try:
        import shap
        contrib = pd.Series(
            shap.TreeExplainer(clf).shap_values(X)[0][:, list(clf.classes_).index("empeoro")],
            index=M.FEATURES)
        titulo = "Cuánto empuja cada factor hacia un empeoramiento"
    except Exception:
        contrib = pd.Series(clf.feature_importances_, index=M.FEATURES)
        titulo = "Factores que más pesan en el modelo (importancia general)"

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)[::-1]
    fig2, ax = plt.subplots(figsize=(8, 2.8))
    ax.barh([M.NOMBRES_LEGIBLES[i] for i in top.index], top.values,
            color=[COLOR_HURTO if v > 0 else "#2a9d8f" for v in top.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(titulo + "\n(rojo: empuja hacia empeorar · verde: empuja hacia mejorar)",
                 fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        f"⚠️ Esta previsión acierta aproximadamente **{metricas['precision_alerta'] * 100:.0f} de "
        "cada 100 veces** que marca un municipio en rojo. Úsala para decidir a quién llamar "
        "primero, no para asignar presupuesto."
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

*Este caso es más común de lo que parece: aplica a 4 de cada 10 municipios del país.*
""")
elif p["cuadrante"] == "bajo gasto / alto hurto":
    st.markdown("""
**Acción recomendada: candidato prioritario a cofinanciación.**

1. Es de los pocos municipios con **inversión baja verificada** y hurto alto — el perfil que
   FONSECON busca, y que sí tiene datos que lo respalden.
2. Revisar si la baja inversión es por falta de recursos propios o por baja capacidad de
   formulación de proyectos: la respuesta cambia entre girar plata y dar asistencia técnica.
3. Comparar contra los municipios similares de la tabla de arriba para dimensionar cuánto sería
   razonable.
""")
elif p["cuadrante"] == "alto gasto / alto hurto":
    st.markdown("""
**Acción recomendada: revisar antes de aumentar el presupuesto.**

1. Invertir más aquí probablemente no cambie el resultado por sí solo: este es el perfil típico de
   **gasto reactivo**, donde la inversión responde al problema en vez de anticiparlo.
2. Revisar la **composición** del gasto (cuánto es vigilancia armada y cuánto otros rubros) y los
   tiempos de adjudicación, antes que el monto total.
3. Es un buen candidato a evaluación cualitativa en terreno: los datos no alcanzan para explicar
   por qué el gasto alto no se traduce en resultados.
""")
elif p["cuadrante"] == "alto gasto / bajo hurto":
    st.markdown("""
**Acción recomendada: revisar eficiencia, con cautela.**

1. Gasta por encima de la mediana y tiene poco hurto. Antes de leerlo como éxito, verificar que no
   sea un municipio pequeño donde **un solo contrato** distorsiona la cifra por habitante.
2. Si la inversión es sostenida en el tiempo, vale la pena documentar qué está haciendo: puede ser
   un caso replicable.
3. También puede indicar sobreinversión relativa. Contrastar con los municipios similares.
""")
else:
    st.markdown("""
**Acción recomendada: seguimiento de rutina.**

1. Inversión y hurto por debajo de la mediana nacional. No es prioritario para asignación ni para
   auditoría en este momento.
2. Vigilar la tendencia: si el hurto empieza a subir de forma sostenida, reevaluar.
""")

st.divider()
st.caption(
    "Prototipo académico — Técnicas de Aprendizaje de Máquina, Pontificia Universidad Javeriana. "
    "Fuentes: SECOP II y Policía Nacional (datos.gov.co), proyecciones de población del DANE. "
    "Las cifras de hurto corresponden a **denuncias registradas**, no al delito real: se estima "
    "que una fracción importante de los hurtos no se denuncia. No usar para decisiones "
    "presupuestales sin validación adicional."
)
