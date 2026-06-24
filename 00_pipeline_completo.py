"""
SCRIPT MAESTRO: pipeline completo de priorizacion de lavado de postes
=========================================================================

Este script conecta los 4 modulos en orden. Es la PLANTILLA que debes
adaptar con tus dataframes reales (los 7 excels que ya cargaste con
pandas). Los pasos marcados con "### AJUSTA AQUI ###" son los que
necesitas modificar para tu caso.

FLUJO:
  1. Cargar tus excels (ya los tienes en pandas)
  2. Modulo 1+2: limpiar IDs y consolidar todas las tablas con id_canonico
  3. Modulo 3: construir el target (velocidad de ensuciamiento + riesgo de corto)
  4. Modulo 4: feature engineering + entrenar modelo + generar ranking final
  5. Exportar resultados (ranking completo + plan de lavado) a Excel

REQUISITOS:
    pip install pandas numpy lightgbm scikit-learn openpyxl
"""

import pandas as pd
import numpy as np
import importlib.util
import os

CARPETA_MODULOS = os.path.dirname(__file__)


def _importar_modulo(nombre_archivo, nombre_modulo):
    spec = importlib.util.spec_from_file_location(
        nombre_modulo, os.path.join(CARPETA_MODULOS, nombre_archivo)
    )
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


limpieza = _importar_modulo("01_limpieza_ids.py", "limpieza_ids")
consolidacion = _importar_modulo("02_consolidacion_tablas.py", "consolidacion_tablas")
target = _importar_modulo("03_construccion_target.py", "construccion_target")
modelo_ml = _importar_modulo("04_modelo_ml.py", "modelo_ml")


def correr_pipeline_completo(
    df_maestro_postes: pd.DataFrame,
    df_lavados: pd.DataFrame,
    df_chequeos: pd.DataFrame,
    df_cortos: pd.DataFrame,
    capacidad_diaria_total: int = 120,
    horizonte_dias_plan: int = 30,
    ruta_salida_excel: str = "resultado_priorizacion_postes.xlsx",
):
    """
    Ejecuta el pipeline completo de principio a fin.

    ### AJUSTA AQUI ###: revisa primero DEFINICION_TABLAS en el archivo
    02_consolidacion_tablas.py para que los nombres de columna coincidan
    con tus excels reales ANTES de correr esta funcion.
    """

    # -----------------------------------------------------------------
    # PASO 1: limpieza de IDs + consolidacion con id_canonico
    # -----------------------------------------------------------------
    print("=" * 70)
    print("PASO 1: Limpieza y consolidacion de identificadores")
    print("=" * 70)

    dataframes_crudos = {
        "maestro_postes": df_maestro_postes,
        "lavados": df_lavados,
        "chequeos": df_chequeos,
        "cortos": df_cortos,
    }

    tablas_consolidadas = consolidacion.consolidar_todas_las_tablas(dataframes_crudos)

    maestro = tablas_consolidadas["maestro_postes"]
    lavados = tablas_consolidadas["lavados"]
    chequeos = tablas_consolidadas["chequeos"]
    cortos = tablas_consolidadas["cortos"]

    # -----------------------------------------------------------------
    # PASO 2: construccion del target (velocidad de ensuciamiento + riesgo)
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 2: Construccion del target de prioridad")
    print("=" * 70)

    eventos_chequeo = target.construir_eventos_chequeo_con_antiguedad_lavado(
        chequeos, lavados, maestro_fechas_instalacion=maestro
    )

    velocidad = target.estimar_velocidad_ensuciamiento_por_poste(eventos_chequeo)
    riesgo_corto = target.construir_senal_riesgo_corto(cortos)
    target_final = target.construir_target_prioridad(velocidad, riesgo_corto)

    # -----------------------------------------------------------------
    # PASO 3: feature engineering + entrenamiento del modelo
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 3: Feature engineering y entrenamiento del modelo")
    print("=" * 70)

    df_features = modelo_ml.construir_features(maestro, target_final)
    df_con_target, df_sin_target = modelo_ml.preparar_dataset_modelo(df_features)

    print(f"\nPostes con historial suficiente (entrenamiento): {len(df_con_target)}")
    print(f"Postes sin historial (el modelo estimara su prioridad): {len(df_sin_target)}")

    if len(df_con_target) < 30:
        print(
            "\nADVERTENCIA: tienes muy pocos postes con historial suficiente "
            "para entrenar (<30). Los resultados del modelo no seran "
            "confiables todavia. Revisa el PASO 1 (es probable que muchos "
            "IDs no esten matcheando) o espera a acumular mas chequeos."
        )

    modelo_entrenado, metricas, importancia = modelo_ml.entrenar_modelo_prioridad(
        df_con_target
    )

    print("\n=== Importancia de variables (que explica el ranking) ===")
    print(importancia.to_string(index=False))

    # -----------------------------------------------------------------
    # PASO 4: prediccion sobre TODOS los postes + plan de lavado
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PASO 4: Prediccion final y plan de lavado")
    print("=" * 70)

    df_completo_con_scores = modelo_ml.predecir_prioridad_para_todos(
        modelo_entrenado, df_features
    )

    plan_lavado = modelo_ml.generar_plan_lavado(
        df_completo_con_scores,
        capacidad_diaria_total=capacidad_diaria_total,
        horizonte_dias=horizonte_dias_plan,
    )

    # -----------------------------------------------------------------
    # PASO 5: exportar resultados a Excel
    # -----------------------------------------------------------------
    columnas_ranking_completo = [
        "id_canonico", "distrito", "tipo_poste", "red_electrica",
        "score_final", "prioridad_1_10_final", "fuente_score",
        "pendiente_ensuciamiento_dia", "dias_desde_ultimo_lavado_actual",
        "n_cortos_bajos", "n_cortos_criticos", "score_riesgo_corto",
    ]
    columnas_ranking_completo = [
        c for c in columnas_ranking_completo if c in df_completo_con_scores.columns
    ]

    ranking_completo = df_completo_con_scores[columnas_ranking_completo].sort_values(
        "score_final", ascending=False
    )

    with pd.ExcelWriter(ruta_salida_excel, engine="openpyxl") as writer:
        ranking_completo.to_excel(writer, sheet_name="ranking_completo", index=False)
        plan_lavado.to_excel(writer, sheet_name="plan_lavado", index=False)
        importancia.to_excel(writer, sheet_name="importancia_features", index=False)

    print(f"\nResultados exportados a: {ruta_salida_excel}")
    print("  - hoja 'ranking_completo': los 40k postes con su score de prioridad")
    print("  - hoja 'plan_lavado': los proximos postes a lavar segun capacidad real")
    print("  - hoja 'importancia_features': que variables explican mas el ranking")

    return {
        "ranking_completo": ranking_completo,
        "plan_lavado": plan_lavado,
        "modelo": modelo_entrenado,
        "metricas": metricas,
        "importancia": importancia,
    }


if __name__ == "__main__":
    print(
        "Este script es una PLANTILLA. Para usarlo con tus datos reales:\n\n"
        "  1. Revisa y ajusta DEFINICION_TABLAS en 02_consolidacion_tablas.py\n"
        "     con los nombres reales de columnas de tus 7 excels.\n"
        "  2. Revisa PATRONES_ID en 01_limpieza_ids.py y agrega los formatos\n"
        "     de codigo de poste que falten (ya tienes 5digitos+letra y\n"
        "     9digitos configurados).\n"
        "  3. Carga tus dataframes reales y llama a correr_pipeline_completo():\n\n"
        "     resultado = correr_pipeline_completo(\n"
        "         df_maestro_postes=tu_df_maestro,\n"
        "         df_lavados=tu_df_lavados,\n"
        "         df_chequeos=tu_df_chequeos,\n"
        "         df_cortos=tu_df_cortos,\n"
        "         capacidad_diaria_total=120,  # AJUSTA: 30 x num. grupos reales\n"
        "         horizonte_dias_plan=30,\n"
        "     )\n"
    )
