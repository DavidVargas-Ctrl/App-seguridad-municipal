# Consulta de Seguridad Municipal — app web (Fase 3)

Prototipo desplegable de la Fase 3 del proyecto. Responde, para un municipio concreto: cómo se
compara su inversión en seguridad con su nivel de hurto, a qué otros municipios se parece, y si el
hurto tiende a empeorar el próximo año.

## Cómo probarla en tu computador

```bash
pip install -r app/requirements.txt
cd app
streamlit run app.py
```

Abre `http://localhost:8501`. La primera carga tarda unos segundos porque entrena el modelo.

## Cómo desplegarla en Streamlit Community Cloud

1. Sube el proyecto a un repositorio de GitHub. **Importante:** `data/raw/` pesa 293 MB y un
   archivo supera el límite de 100 MB de GitHub — añade `data/` al `.gitignore`. La carpeta
   `app/datos/` (2,3 MB) **sí** debe subirse, porque es lo que la app consume.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **New app** → selecciona el repositorio y la rama.
4. En *Main file path* escribe `app/app.py`.
5. **Deploy**. El primer despliegue tarda 2-3 minutos instalando dependencias.

La alternativa es Hugging Face Spaces: crear un Space de tipo *Streamlit*, subir el contenido de
`app/` en la raíz del Space y renombrar `app.py` si la plataforma lo pide.

## Estructura

| Archivo | Qué hace |
|---|---|
| `app.py` | Interfaz: selector de municipio, indicadores, gráfica, previsión, explicación y acciones recomendadas |
| `modelo.py` | Pipeline de datos y modelo. Replica la Parte 4 del cuaderno: mismas features, mismo corte temporal, mismos cortes de clase |
| `datos/dataset_municipio_anio_final.csv` | Panel municipio-año que genera el cuaderno |
| `datos/municipios.csv` | Códigos DIVIPOLA con nombre de municipio y departamento |

El modelo **se reentrena al arrancar** en vez de cargarse desde un `pickle`. Entrenar un Random
Forest sobre 5.565 filas tarda menos de dos segundos, y así la app no se rompe si la versión de
scikit-learn del servidor no coincide con la que generó el archivo serializado.

Para regenerar los datos tras un cambio en el cuaderno:

```bash
cp data/processed/dataset_municipio_anio_final.csv app/datos/
```

## Decisiones de diseño

La app está hecha para un funcionario, no para un científico de datos. Tres decisiones se
desprenden directamente de los hallazgos del análisis:

**Muestra la cobertura de datos antes que cualquier cifra.** 485 de 1.110 municipios no tienen
ningún contrato registrado en SECOP. En el análisis, el 93% de la lista de municipios
"prioritarios" resultó ser municipios de los que no se sabe cuánto gastan, no municipios que
gastan poco. Por eso la app nunca muestra "inversión: $0" — muestra "sin dato" y explica la
diferencia, y las acciones recomendadas para esos municipios son de **auditoría de reporte**, no
de asignación de recursos.

**Dice qué tan mala es la alerta.** De cada 100 municipios que el modelo marca en rojo, acierta
unos 23. Esa cifra aparece junto a la predicción, no escondida en una nota al pie, porque
presentar una precisión de 0,23 como "predicción" ante quien asigna presupuesto sería engañoso.
La app la presenta como filtro de atención.

**Bloquea la lectura causal.** El hallazgo más repetible del proyecto es que gasto y hurto
correlacionan *positivamente*. Leído a la ligera diría "la vigilancia aumenta el hurto", lo cual es
falso: refleja gasto reactivo. La sección de contexto lo advierte explícitamente.
