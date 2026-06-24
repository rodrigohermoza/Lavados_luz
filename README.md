# Pipeline de priorizacion de lavado de postes

## Que resuelve esto

No tienes "fallas" como evento limpio. Tienes 3 señales indirectas:
chequeos (que tan sucio se ve un poste), lavados (cuando se limpio), y
cortos bajos/criticos (que pueden o no deberse a suciedad). Este pipeline
convierte eso en un score de prioridad 1-10 por poste, defendible y
explicable, listo para generar un plan de lavado bajo la restriccion real
de capacidad (30 lavados/dia x N grupos).

## Por que el target se construye asi (y no se inventa a mano)

1. **Velocidad de ensuciamiento**: para cada poste, se mide cuantos dias
   tarda en pasar de "recien lavado" a "sucio" segun sus propios chequeos
   historicos (regresion simple dias -> nivel de suciedad).
2. **Riesgo de corto**: se pondera el historial de cortos (mas peso a
   criticos y a los mas recientes) como factor que AMPLIFICA la prioridad,
   sin asumir causalidad directa.
3. **Combinacion + escala 1-10 por percentil**: el score final es
   0.7 x velocidad + 0.3 x riesgo_corto (ajustable), convertido a deciles
   sobre el universo real de postes -- no umbrales arbitrarios.
4. **Modelo de ML (LightGBM)**: como no todos los 40k postes tendran
   chequeos suficientes, se entrena un modelo con los que SI tienen
   historial, y se usa para estimar la prioridad de los que no, a partir
   de sus caracteristicas (distrito, tipo, antiguedad, etc).

## Orden de archivos

| Archivo | Que hace |
|---|---|
| `01_limpieza_ids.py` | Normaliza identificadores de poste (errores de tipeo, OCR, distintos formatos) |
| `02_consolidacion_tablas.py` | Cruza tus 7 excels usando un `id_canonico` comun |
| `03_construccion_target.py` | Construye el target de prioridad desde chequeos + lavados + cortos |
| `04_modelo_ml.py` | Feature engineering, entrenamiento LightGBM, ranking final, plan de lavado |
| `00_pipeline_completo.py` | Conecta todo. Es el que ejecutas con tus datos reales |

## Pasos para usarlo con tus datos reales

### 1. Ajusta `01_limpieza_ids.py`
Revisa `PATRONES_ESPECIFICOS`. Ya tiene configurado:
- 5 digitos + 1 letra (`12345A`)
- 9 digitos (`123456789`)

Si tienes mas formatos de codigo de poste, agregalos ahi siguiendo el
mismo patron `(nombre, regex, solo_digitos_esperado)`.

### 2. Ajusta `02_consolidacion_tablas.py`
Edita `DEFINICION_TABLAS` con los nombres REALES de:
- las columnas candidatas a ID en cada uno de tus 7 excels
- las columnas utiles que quieras conservar (distrito, fecha, etc.)

Corre esto primero por separado y revisa el "reporte de calidad" y el
"% sin match" de cada tabla. Si el % sin match es alto (>5-10%), antes de
seguir conviene investigar por que (ver la nota sobre join espacial al
final de ese archivo).

### 3. Corre el pipeline completo
```python
import pandas as pd
from postes_ml import pipeline  # o importa 00_pipeline_completo.py directo

resultado = pipeline.correr_pipeline_completo(
    df_maestro_postes=tu_df_maestro,
    df_lavados=tu_df_lavados,
    df_chequeos=tu_df_chequeos,
    df_cortos=tu_df_cortos,
    capacidad_diaria_total=120,   # AJUSTA: 30 x numero real de grupos
    horizonte_dias_plan=30,
    ruta_salida_excel="ranking_postes.xlsx",
)
```

Esto genera un Excel con 3 hojas:
- **ranking_completo**: los 40k postes con su score y deciles 1-10
- **plan_lavado**: los proximos N postes a lavar, respetando capacidad real
- **importancia_features**: que variables explican mas el ranking (util
  para la reunion con tu jefe)

## Cosas para revisar / decisiones que tomaras tu

- **Pesos del target** (`peso_velocidad=0.7, peso_riesgo_corto=0.3` en
  `03_construccion_target.py`): ajustalos si el negocio prioriza mas
  evitar cortos que la suciedad pura.
- **Mapeo de niveles de suciedad/corto a numeros**: revisa
  `mapear_nivel_suciedad_a_score` y `peso_nivel` en `03_construccion_target.py`
  si tus categorias reales tienen otros nombres.
- **Columnas de agrupamiento para imputar pendiente faltante**
  (`columnas_grupo=["distrito", "tipo_poste"]` en `04_modelo_ml.py`):
  puedes anadir mas (ej. red_electrica) si tiene sentido en tu negocio.
- **MAE de cross-validation**: el pipeline imprime esto automaticamente.
  Comparalo siempre contra el "MAE baseline" (modelo que predice el
  promedio) -- si tu modelo no le gana por bastante margen, revisa las
  features antes de confiar en el ranking.

## Siguientes fases (segun lo que mencionaste que pide tu jefe)

Este pipeline cubre la Fase 1 (priorizacion + plan basico). Cuando avances:
- **Clustering para rutas optimas**: usa `latitud`/`longitud` del maestro
  + el score de prioridad para clustering geografico (ej. K-Means o
  DBSCAN) y resolver como un problema de ruteo (VRP) sobre los postes ya
  priorizados.
- **Costos de riesgo en dinero**: cuando tengas costos de reparacion /
  interrupcion, multiplica `score_riesgo_corto` por el costo esperado en
  soles para traducir el ranking a perdida economica evitada.
- **Mapa de coordenadas**: el `ranking_completo` ya incluye todo lo
  necesario para plotear en un mapa (folium, plotly, o un GIS real) una
  vez que conectes lat/long.
