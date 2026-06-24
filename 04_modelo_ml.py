"""
MODULO 4: Feature engineering + entrenamiento del modelo de ML
==================================================================

OBJETIVO DE ESTE MODULO:
1. Unir maestro de postes + target (modulo 3) + features derivadas.
2. Imputar la pendiente de ensuciamiento para postes con <2 chequeos
   usando el promedio de su grupo (mismo tipo de poste + distrito),
   en vez de dejarlos fuera del modelo o en cero (lo cual subestimaria
   su prioridad).
3. Entrenar un modelo de regresion (LightGBM) que prediga el
   score_prioridad_0_1 para CUALQUIER poste, incluyendo los que no
   tienen historial suficiente de chequeos -- esto es clave porque
   con 40k postes, muchos tendran poco o ningun historial, y el
   modelo debe poder generalizar a partir de sus caracteristicas
   (tipo, distrito, antiguedad, dias desde ultimo lavado) aunque no
   tengan chequeos propios.
4. Generar el ranking final 1-10 + un ranking continuo (mejor para
   ordenar bajo restriccion de capacidad de 30 lavados/dia/grupo).

POR QUE LIGHTGBM:
- Maneja bien variables categoricas (distrito, tipo_poste, red_electrica)
  sin necesidad de one-hot encoding manual.
- Maneja bien valores faltantes nativamente (no hace falta imputar TODO).
- Es rapido incluso con 40k+ filas y se entrena en segundos.
- Da importancia de variables (feature importance), util para explicarle
  a tu jefe QUE esta impulsando el ranking (ej. "el distrito X y la
  antiguedad del poste explican el 60% del riesgo").
"""

import pandas as pd
import numpy as np
from datetime import datetime

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# 1. IMPUTACION DE PENDIENTE POR GRUPO (para postes con poco historial)
# ---------------------------------------------------------------------------

def imputar_pendiente_por_grupo(
    df: pd.DataFrame,
    columnas_grupo: list = None,
    col_pendiente: str = "pendiente_ensuciamiento_dia",
) -> pd.DataFrame:
    """
    Para postes donde 'pendiente_ensuciamiento_dia' es NaN (porque
    tuvieron <2 chequeos utiles), imputa con el promedio del grupo al
    que pertenecen (ej. mismo distrito + mismo tipo de poste).

    Si el grupo completo no tiene ningun dato (caso raro, distrito o
    tipo nuevo sin historial), se usa el promedio GLOBAL como ultimo
    recurso, y se marca con 'pendiente_imputada_global' = True para que
    sepas que ese numero es menos confiable.

    AJUSTA columnas_grupo segun lo que tenga mas sentido en tu negocio.
    Por defecto usamos distrito + tipo_poste porque son los factores
    mas intuitivos de velocidad de ensuciamiento (zona + material/diseño).
    """
    if columnas_grupo is None:
        columnas_grupo = ["distrito", "tipo_poste"]

    df = df.copy()
    df["pendiente_imputada"] = df[col_pendiente].isna()

    columnas_grupo_existentes = [c for c in columnas_grupo if c in df.columns]
    if not columnas_grupo_existentes:
        # No hay columnas de grupo disponibles: usar promedio global directo
        promedio_global = df[col_pendiente].mean()
        df[col_pendiente] = df[col_pendiente].fillna(promedio_global)
        df["pendiente_imputada_global"] = df["pendiente_imputada"]
        return df

    promedio_por_grupo = df.groupby(columnas_grupo_existentes)[col_pendiente].transform("mean")
    df[col_pendiente] = df[col_pendiente].fillna(promedio_por_grupo)

    # Si despues de imputar por grupo TODAVIA hay NaN (grupo sin ningun dato),
    # usar el promedio global como ultimo recurso
    promedio_global = df[col_pendiente].mean()
    df["pendiente_imputada_global"] = df[col_pendiente].isna()
    df[col_pendiente] = df[col_pendiente].fillna(promedio_global)

    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def construir_features(
    maestro: pd.DataFrame,
    target_y_velocidad: pd.DataFrame,
    fecha_referencia: pd.Timestamp = None,
    col_id: str = "id_canonico",
) -> pd.DataFrame:
    """
    Construye la tabla final de features (X) + target (y) por poste,
    lista para entrenar.

    Features incluidas (AJUSTA/EXPANDE segun lo que tengas disponible):
    - tipo_poste, distrito, red_electrica (categoricas)
    - antiguedad_dias: dias desde instalacion hasta hoy
    - dias_desde_ultimo_lavado_actual: dias desde el lavado MAS RECIENTE
      conocido hasta la fecha de referencia (hoy, o la fecha en que vas
      a generar el ranking) -- esta es probablemente la feature MAS
      IMPORTANTE, porque es la "deuda de lavado" actual del poste.
    - pendiente_ensuciamiento_dia (imputada si hace falta)
    - n_cortos_bajos, n_cortos_criticos, score_riesgo_corto
    - n_chequeos_usados: cuantos chequeos respaldan la pendiente estimada
      (sirve como proxy de confianza del dato)
    """
    if fecha_referencia is None:
        fecha_referencia = pd.Timestamp.now()

    df = maestro.merge(target_y_velocidad, on=col_id, how="left")

    if "fecha_instalacion" in df.columns:
        df["antiguedad_dias"] = (
            fecha_referencia - pd.to_datetime(df["fecha_instalacion"])
        ).dt.days
    else:
        df["antiguedad_dias"] = np.nan

    # dias_desde_lavado_en_ultimo_chequeo viene del modulo 3, pero si quieres
    # la "deuda actual" (no solo hasta el ultimo chequeo), idealmente cruza
    # tambien la fecha del lavado MAS RECIENTE real (no solo el que hubo
    # antes del ultimo chequeo). Aqui se deja el campo ya calculado del
    # modulo 3 como aproximacion; si tienes la fecha de ultimo lavado real
    # por poste, reemplaza esta linea por ese calculo directo.
    if "dias_desde_lavado_en_ultimo_chequeo" in df.columns:
        df["dias_desde_ultimo_lavado_actual"] = df["dias_desde_lavado_en_ultimo_chequeo"]
    else:
        df["dias_desde_ultimo_lavado_actual"] = np.nan

    df = imputar_pendiente_por_grupo(df)

    df["n_cortos_bajos"] = df.get("n_cortos_bajos", 0).fillna(0)
    df["n_cortos_criticos"] = df.get("n_cortos_criticos", 0).fillna(0)
    df["score_riesgo_corto"] = df.get("score_riesgo_corto", 0.0).fillna(0.0)
    df["n_chequeos_usados"] = df.get("n_chequeos_usados", 0).fillna(0)

    return df


COLUMNAS_CATEGORICAS = ["tipo_poste", "distrito", "red_electrica"]

COLUMNAS_NUMERICAS = [
    "antiguedad_dias",
    "dias_desde_ultimo_lavado_actual",
    "pendiente_ensuciamiento_dia",
    "n_cortos_bajos",
    "n_cortos_criticos",
    "score_riesgo_corto",
    "n_chequeos_usados",
]

COLUMNA_TARGET = "score_prioridad_0_1"


# ---------------------------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------------------------

def preparar_dataset_modelo(df: pd.DataFrame):
    """
    Separa el dataframe completo en:
    - df_con_target: filas que SI tienen target (postes con al menos
      1 chequeo historico) -> se usan para ENTRENAR.
    - df_sin_target: filas SIN target (postes sin ningun chequeo nunca)
      -> el modelo entrenado se usara para PREDECIR su prioridad.

    Tambien convierte las columnas categoricas a tipo 'category' (requerido
    por LightGBM para manejarlas nativamente sin one-hot encoding).
    """
    df = df.copy()
    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    tiene_target = df[COLUMNA_TARGET].notna()
    df_con_target = df[tiene_target].copy()
    df_sin_target = df[~tiene_target].copy()

    return df_con_target, df_sin_target


def entrenar_modelo_prioridad(
    df_con_target: pd.DataFrame,
    n_folds: int = 5,
    semilla: int = 42,
):
    """
    Entrena un modelo LightGBM de REGRESION para predecir score_prioridad_0_1
    a partir de las features definidas arriba.

    Usa K-Fold cross-validation (no un simple train/test split) porque
    con un numero moderado de postes con historial suficiente, queremos
    aprovechar todos los datos para validar de forma robusta.

    Devuelve:
    - modelo_final: entrenado con TODOS los datos disponibles (para usar
      en produccion / prediccion sobre los postes sin target)
    - metricas_cv: MAE y RMSE promedio en cross-validation (para reportar
      a tu jefe que tan preciso es el modelo)
    - importancia_features: dataframe con la importancia de cada variable
    """
    columnas_features = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS
    columnas_features = [c for c in columnas_features if c in df_con_target.columns]

    X = df_con_target[columnas_features]
    y = df_con_target[COLUMNA_TARGET]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=semilla)

    maes, rmses = [], []
    for fold_i, (idx_train, idx_val) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
        y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]

        modelo_fold = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            random_state=semilla,
            verbosity=-1,
        )
        modelo_fold.fit(
            X_train, y_train,
            categorical_feature=[c for c in COLUMNAS_CATEGORICAS if c in X_train.columns],
        )

        pred_val = modelo_fold.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred_val))
        rmses.append(np.sqrt(mean_squared_error(y_val, pred_val)))

        print(f"Fold {fold_i}/{n_folds} -> MAE: {maes[-1]:.4f}  RMSE: {rmses[-1]:.4f}")

    print(f"\nMAE promedio (cross-validation): {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
    print(f"RMSE promedio (cross-validation): {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
    print(
        "\nInterpretacion: el score_prioridad_0_1 va de 0 a 1. Un MAE de "
        f"{np.mean(maes):.3f} significa que, en promedio, el modelo se "
        "equivoca por esa magnitud en la escala 0-1 al estimar la "
        "prioridad de un poste. Compara esto contra simplemente usar el "
        "promedio general como base (ver linea de comparacion abajo)."
    )

    # Baseline de comparacion: que tan mal lo haria un modelo "tonto" que
    # siempre predice el promedio. Si tu modelo no le gana por mucho a
    # esto, hay que revisar las features.
    mae_baseline = mean_absolute_error(y, np.full_like(y, y.mean(), dtype=float))
    print(f"MAE si siempre predijeramos el promedio (baseline): {mae_baseline:.4f}")

    # Entrenar el modelo FINAL con todos los datos disponibles
    modelo_final = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        random_state=semilla,
        verbosity=-1,
    )
    modelo_final.fit(
        X, y,
        categorical_feature=[c for c in COLUMNAS_CATEGORICAS if c in X.columns],
    )

    importancia_features = pd.DataFrame({
        "feature": columnas_features,
        "importancia": modelo_final.feature_importances_,
    }).sort_values("importancia", ascending=False)

    metricas_cv = {
        "mae_promedio": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_promedio": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "mae_baseline": mae_baseline,
    }

    return modelo_final, metricas_cv, importancia_features


def predecir_prioridad_para_todos(
    modelo,
    df_completo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Usa el modelo entrenado para predecir score_prioridad_0_1 sobre TODOS
    los postes (incluyendo los que no tenian chequeos historicos, que
    son justamente los que mas necesitan ser estimados por el modelo en
    vez de calculados directo de datos).

    Devuelve el dataframe completo con columnas nuevas:
    - score_prioridad_predicho: 0 a 1
    - prioridad_1_10_predicha: por percentil sobre TODOS los 40k postes
    - fuente_score: 'historico' (calculado directo del modulo 3) o
      'modelo' (estimado porque no tenia historial suficiente)
    """
    columnas_features = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS
    columnas_features = [c for c in columnas_features if c in df_completo.columns]

    df = df_completo.copy()
    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df["score_prioridad_predicho"] = modelo.predict(df[columnas_features])

    # Para los postes que SI tenian score historico real, preferimos usar
    # ese valor (mas confiable que la prediccion) y dejamos el modelo solo
    # para rellenar donde no habia informacion directa.
    df["fuente_score"] = np.where(
        df[COLUMNA_TARGET].notna(), "historico", "modelo"
    )
    df["score_final"] = df[COLUMNA_TARGET].fillna(df["score_prioridad_predicho"])

    df["prioridad_1_10_final"] = (
        pd.qcut(df["score_final"], 10, labels=False, duplicates="drop") + 1
    )

    return df


# ---------------------------------------------------------------------------
# 4. RANKING FINAL BAJO RESTRICCION DE CAPACIDAD
# ---------------------------------------------------------------------------

def generar_plan_lavado(
    df_con_scores: pd.DataFrame,
    capacidad_diaria_total: int = 120,  # ~30 x 4 grupos, AJUSTA a tu caso real
    horizonte_dias: int = 30,
    col_id: str = "id_canonico",
) -> pd.DataFrame:
    """
    Convierte el ranking de prioridad en un PLAN concreto: cuales postes
    se deben lavar en los proximos `horizonte_dias` dias, respetando la
    capacidad operativa real (capacidad_diaria_total * horizonte_dias).

    Esto es lo que tu jefe probablemente quiere ver primero: no solo
    "estos son los mas sucios" sino "esto es lo que hay que hacer esta
    semana/mes con los recursos que tenemos".
    """
    capacidad_total = capacidad_diaria_total * horizonte_dias

    plan = (
        df_con_scores.sort_values("score_final", ascending=False)
        .head(capacidad_total)
        .copy()
        .reset_index(drop=True)
    )
    plan["orden_prioridad"] = np.arange(1, len(plan) + 1)
    plan["dia_estimado_de_lavado"] = ((plan["orden_prioridad"] - 1) // capacidad_diaria_total) + 1

    print(
        f"\nPlan generado: {len(plan)} postes a lavar en los proximos "
        f"{horizonte_dias} dias ({capacidad_diaria_total} postes/dia)."
    )
    print(
        f"Esto representa el {len(plan) / len(df_con_scores) * 100:.1f}% "
        f"del total de {len(df_con_scores)} postes."
    )

    return plan[[col_id, "score_final", "prioridad_1_10_final", "fuente_score",
                 "orden_prioridad", "dia_estimado_de_lavado"]]


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EJEMPLO SINTETICO de extremo a extremo, con suficientes postes para
    # que LightGBM tenga algo razonable que aprender. Sustituye por tus
    # datos reales conectando con los modulos 1-3.
    # ------------------------------------------------------------------
    np.random.seed(42)
    n_postes = 500

    distritos = np.random.choice(["Surco", "Miraflores", "SJL", "Comas", "Chorrillos"], n_postes)
    tipos = np.random.choice(["concreto", "metal", "fibra"], n_postes)

    maestro = pd.DataFrame({
        "id_canonico": [f"P{i:05d}" for i in range(n_postes)],
        "distrito": distritos,
        "tipo_poste": tipos,
        "red_electrica": np.random.choice(["RED-01", "RED-02", "RED-03"], n_postes),
        "fecha_instalacion": pd.to_datetime("2015-01-01") + pd.to_timedelta(
            np.random.randint(0, 3000, n_postes), unit="D"
        ),
    })

    # Simulamos que la "verdad oculta" depende del distrito (costero ensucia
    # mas rapido) + tipo de poste, para que el modelo tenga patrones reales
    # que aprender, similar a lo que pasaria con datos reales de clima/zona.
    factor_distrito = maestro["distrito"].map({
        "Chorrillos": 1.8, "Miraflores": 1.5, "Surco": 1.0, "SJL": 0.8, "Comas": 0.9
    })
    pendiente_real = factor_distrito * np.random.uniform(0.001, 0.01, n_postes)

    # Solo el 60% de postes tiene historial de chequeos suficiente (simula
    # la realidad: no todos los 40k postes tendran chequeos historicos)
    tiene_historial = np.random.rand(n_postes) < 0.6

    target_velocidad = pd.DataFrame({
        "id_canonico": maestro["id_canonico"],
        "pendiente_ensuciamiento_dia": np.where(tiene_historial, pendiente_real, np.nan),
        "n_chequeos_usados": np.where(tiene_historial, np.random.randint(2, 8, n_postes), 0),
        "dias_desde_lavado_en_ultimo_chequeo": np.random.randint(5, 200, n_postes),
        "score_riesgo_corto": np.random.beta(2, 5, n_postes),
        "n_cortos_bajos": np.random.poisson(0.5, n_postes),
        "n_cortos_criticos": np.random.poisson(0.1, n_postes),
    })

    # El target final solo existe (no-NaN) para los que tienen historial
    score_base = (
        0.7 * pendiente_real.rank(pct=True) + 0.3 * target_velocidad["score_riesgo_corto"]
    )
    target_velocidad["score_prioridad_0_1"] = np.where(tiene_historial, score_base, np.nan)

    df_features = construir_features(maestro, target_velocidad)

    df_con_target, df_sin_target = preparar_dataset_modelo(df_features)
    print(f"Postes con historial (entrenamiento): {len(df_con_target)}")
    print(f"Postes sin historial (a predecir): {len(df_sin_target)}")

    modelo, metricas, importancia = entrenar_modelo_prioridad(df_con_target)

    print("\n=== Importancia de features ===")
    print(importancia)

    df_completo_con_scores = predecir_prioridad_para_todos(modelo, df_features)

    print("\n=== Muestra de resultados finales ===")
    print(df_completo_con_scores[[
        "id_canonico", "distrito", "tipo_poste", "score_final",
        "prioridad_1_10_final", "fuente_score"
    ]].sort_values("score_final", ascending=False).head(10))

    plan = generar_plan_lavado(df_completo_con_scores, capacidad_diaria_total=30, horizonte_dias=10)
    print("\n=== Plan de lavado (primeros 10 dias, 30/dia) ===")
    print(plan.head(15))
