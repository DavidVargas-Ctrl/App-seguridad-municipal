# Consulta de Seguridad Municipal — Fase 3

Aplicación web desarrollada para la Fase 3 del proyecto. Permite consultar un municipio y revisar su inversión en seguridad, sus niveles de hurto, municipios con características similares y la estimación del comportamiento del hurto para el siguiente año.

## Estructura del proyecto

| Archivo                                  | Descripción                                                                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                                 | Contiene la interfaz de la aplicación, el selector de municipio, los indicadores, las gráficas, la predicción y las recomendaciones.                                      |
| `modelo.py`                              | Contiene el procesamiento de datos y el modelo utilizado para generar las predicciones. Mantiene las variables y el corte temporal utilizados en la Parte 4 del notebook. |
| `datos/dataset_municipio_anio_final.csv` | Dataset final organizado por municipio y año.                                                                                                                             |
| `datos/municipios.csv`                   | Contiene los códigos DIVIPOLA junto con el nombre del municipio y departamento.                                                                                           |

El modelo se vuelve a entrenar cada vez que se inicia la aplicación. Se tomó esta decisión debido a que el entrenamiento del Random Forest con 5.565 registros tarda menos de dos segundos y permite evitar problemas de compatibilidad entre versiones de `scikit-learn` al utilizar archivos serializados.

En caso de modificar el procesamiento realizado en el notebook, los datos utilizados por la aplicación se pueden actualizar mediante:

```bash
cp data/processed/dataset_municipio_anio_final.csv app/datos/
```

## Decisiones de la aplicación

La aplicación está orientada principalmente a facilitar la consulta de los resultados por parte de funcionarios o personas que no necesariamente trabajan directamente con modelos de datos.

### Cobertura de los datos

Uno de los principales problemas encontrados durante el análisis fue la falta de información para varios municipios. De los 1.110 municipios analizados, 485 no presentan contratos registrados en SECOP dentro de los datos utilizados.

Además, aproximadamente el 93 % de los municipios que inicialmente aparecían como prioritarios correspondían en realidad a municipios sin información suficiente sobre inversión.

Por esta razón, cuando no existe información de inversión, la aplicación muestra **“sin dato”** en lugar de asumir que la inversión fue de $0. Para estos casos se recomienda revisar primero la disponibilidad y calidad de la información antes de obtener conclusiones sobre la asignación de recursos.

### Interpretación de las alertas

El modelo no se plantea como una predicción definitiva del comportamiento del hurto. Entre los municipios identificados con una alerta alta, aproximadamente 23 de cada 100 corresponden efectivamente a casos positivos.

Por esta razón, la predicción se presenta como una herramienta para identificar municipios que podrían requerir una revisión adicional y no como un criterio único para tomar decisiones presupuestales.

### Relación entre inversión y hurto

Durante el análisis se encontró una correlación positiva entre el gasto en seguridad y los niveles de hurto.

Este resultado no implica que una mayor inversión produzca un aumento del hurto. Una posible explicación es que parte del gasto sea reactivo, es decir, que los municipios con mayores problemas de seguridad destinen más recursos para responder a estas situaciones.

Por esta razón, la aplicación no presenta esta relación como causal y la utiliza únicamente como parte del contexto del análisis.

