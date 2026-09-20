# Consulta de Seguridad Municipal — app web (Fase 3)

Prototipo desplegable de la Fase 3 del proyecto. Responde, para un municipio y un **tipo de
hurto** concretos: cómo se compara su inversión en seguridad con su nivel de ese hurto, a qué
otros municipios se parece, y si ese hurto tiende a empeorar el próximo año.

## ⚠️ Paso 1 antes de nada: actualizar los datos

La app trae el CSV de la versión anterior del panel para poder arrancar sin configuración, pero
**ese archivo no tiene las categorías nuevas** (residencias, comercio, compendio total) ni los
factores de contexto nuevos (capturas, homicidios). Si lo dejas así, la app arranca en modo
reducido y muestra un aviso rojo en pantalla.

Para habilitar las cinco categorías, corre el cuaderno hasta la sección II.4 y copia su salida:

```bash
cp data/processed/dataset_municipio_anio_final.csv app/datos/
```

Ese CSV debe traer, además de lo que ya traía: `hurto_total`, `hurto_residencias_total`,
`hurto_comercio_total`, `capturas_total` y `homicidios_total`. Las tasas per cápita y el
compendio se recalculan solos si faltan.

## Cómo probarla en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre `http://localhost:8501`. La primera carga tarda unos 15 segundos porque entrena el modelo de
la categoría seleccionada y calienta el explicador SHAP. Cambiar de categoría vuelve a costar unos
segundos la primera vez; después queda en caché.

## Cómo desplegarla en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub. **Importante:** `data/raw/` pesa cientos de MB y
   algún archivo supera el límite de 100 MB de GitHub — añade `data/` al `.gitignore`. La carpeta
   `datos/` de la app (2,5 MB) **sí** debe subirse, porque es lo que la app consume.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **New app** → selecciona el repositorio y la rama.
4. En *Main file path* escribe la ruta de `app.py`.
5. **Deploy**. El primer despliegue tarda 2-3 minutos instalando dependencias.

La alternativa es Hugging Face Spaces: crear un Space de tipo *Streamlit* y subir el contenido de
esta carpeta en la raíz del Space.

## Estructura

| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz: selector de tipo de hurto y de municipio, indicadores, gráfica, previsión, explicación, comparador de categorías y acciones recomendadas |
| `modelo.py` | Pipeline de datos y modelos. Replica la Parte 4 rediseñada del cuaderno: cinco categorías, mismas features, mismo corte temporal, mismos cortes de clase |
| `datos/dataset_municipio_anio_final.csv` | Panel municipio-año que genera el cuaderno |
| `datos/municipios.csv` | Códigos DIVIPOLA con nombre de municipio y departamento |

Los modelos **se reentrenan al arrancar** en vez de cargarse desde un `pickle`, y solo se entrena
la categoría que el usuario está mirando. Así la app no se rompe si la versión de scikit-learn del
servidor no coincide con la que generó el archivo serializado.

## Qué cambió respecto a la versión anterior de la app

La app anterior respondía una sola pregunta —hurto a personas— porque esa era la única variable de
resultado que modelaba el cuaderno. La Parte 4 se rehizo para resolver el mismo problema **cinco
veces en paralelo**, y la app recoge ese cambio:

| | Versión anterior | Esta versión |
|---|---|---|
| Variable de resultado | Hurto a personas | 5 categorías: Personas, Vehículos, Residencias, Comercio y el compendio Total |
| Features de contexto | Hurto de vehículos, estupefacientes | Homicidios, estupefacientes y **capturas**; las categorías de hurto salieron |
| Algoritmo | Random Forest para todo | El ganador por F1 macro de cada categoría: Gradient Boosting en 4, Random Forest en Comercio |
| Explicabilidad | SHAP `TreeExplainer` | `TreeExplainer` donde sirve, `Permutation` donde no (ver abajo) |

**Regla de exclusión (la decisión de diseño de esta ronda).** El modelo de una categoría no usa
las demás categorías de hurto como predictor. La única excepción es el nivel actual de la propia
categoría, que se mantiene como control: sin él no se puede distinguir "gasto reactivo" de "gasto
que sí funcionó". Predecir hurto con hurto daría métricas bonitas y ninguna utilidad para FONSECON,
que no controla el hurto sino el presupuesto.

## Decisiones de diseño

La app está hecha para un funcionario, no para un científico de datos. Cuatro decisiones se
desprenden directamente de los hallazgos del análisis:

**Muestra la cobertura de datos antes que cualquier cifra.** 732 de 1.110 municipios no tienen
ningún contrato registrado en SECOP en todo el período — y el cuaderno confirmó que ese patrón se
repite idéntico en las cinco categorías, no es una particularidad de ninguna. Por eso la app nunca
muestra "inversión: $0": muestra "sin dato" y explica la diferencia, y las acciones recomendadas
para esos municipios son de **auditoría de reporte**, no de asignación de recursos.

**Dice qué tan mala es la alerta, y que no es igual de mala en todas las categorías.** Residencias
y comercio se anticipan bastante mejor que el agregado; el compendio total, que es la cifra más
cómoda para comunicar, es la **peor** para predecir. El comparador de categorías pone esa tabla a
la vista en vez de esconder la diferencia detrás de un solo número, porque elegir "hurto total"
para comunicar cuesta poder predictivo y el stakeholder debería saberlo.

**Bloquea la lectura causal.** El hallazgo más repetible del proyecto es que gasto y hurto
correlacionan *positivamente*. Leído a la ligera diría "la vigilancia aumenta el hurto", lo cual es
falso: refleja gasto reactivo. La sección de contexto lo advierte explícitamente.

**Corrige el cuadrante degenerado.** El cuaderno calcula el umbral de "gasto alto" con la mediana
de *todos* los municipios; como 732 de 1.110 tienen gasto 0, esa mediana es exactamente 0 y la
condición "gasto ≥ mediana" se vuelve cierta para todo el mundo — de ahí el reparto 555/555 de la
sección II.5, con los cuadrantes de "bajo gasto" vacíos. Ese 555/555 es un artefacto de cobertura,
no un hallazgo. La app calcula el umbral **solo entre los municipios que sí reportan** y saca a los
que no reportan a una categoría propia, "sin dato de gasto", que es lo que realmente son.

## Nota técnica: por qué el explicador no es siempre SHAP `TreeExplainer`

`shap.TreeExplainer` no soporta `GradientBoostingClassifier` multiclase (lanza
`InvalidModelError`), y Gradient Boosting es el algoritmo ganador en 4 de las 5 categorías. La
salida fácil habría sido caer a `feature_importances_`, pero eso es una importancia **global**: no
explica el municipio concreto que el funcionario tiene en pantalla, que es justamente para lo que
sirve esa gráfica. En su lugar se usa el `Permutation` explainer de SHAP, que es agnóstico al
modelo y sí da contribuciones locales. Cuesta unos segundos la primera vez por compilación interna
de la librería, así que `crear_explicador()` hace una llamada de calentamiento durante la carga:
que ese costo lo pague el spinner y no el primer municipio consultado.
