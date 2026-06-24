"""
MODULO 2: Consolidacion de las tablas en un maestro unico de postes
======================================================================

PROBLEMA QUE RESUELVE:
Tienes ~7 excels distintos (maestro de postes, lavados, chequeos, cortos,
etc.) y cada uno puede tener el identificador del poste en una columna con
nombre distinto, y con calidad distinta. Antes de poder cruzar esta
informacion necesitas:

1. Decidir, para cada tabla, cual es la columna que realmente identifica
   al poste (puede haber varias candidatas).
2. Normalizar esa columna con el modulo 1.
3. Resolver casos donde, tras normalizar, el mismo poste fisico aparece
   con mas de un ID valido en distintas filas/tablas (deduplicacion).
4. Construir una tabla "puente" (id_canonico) que te permita unir
   lavados + chequeos + cortos + maestro sin perder filas por
   inconsistencias de formato.

IMPORTANTE: Ajusta DEFINICION_TABLAS con los nombres reales de tus 7
archivos/dataframes y las columnas candidatas a ID en cada uno.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

import importlib.util
import os

# Importar el modulo 1 (limpieza de IDs). Ajusta el path si organizas
# tus archivos de otra forma en tu entorno de trabajo.
_spec = importlib.util.spec_from_file_location(
    "limpieza_ids", os.path.join(os.path.dirname(__file__), "01_limpieza_ids.py")
)
limpieza_ids = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(limpieza_ids)

limpiar_columna_id = limpieza_ids.limpiar_columna_id
reporte_calidad_ids = limpieza_ids.reporte_calidad_ids


# ---------------------------------------------------------------------------
# 1. CONFIGURACION: AJUSTA ESTO CON TUS TABLAS REALES
# ---------------------------------------------------------------------------
# Para cada tabla, lista las columnas que PODRIAN contener el ID del poste.
# Si hay varias (ej. "codigo_poste" Y "numero_poste_alterno"), el pipeline
# las normaliza todas y usa la primera no-nula como ID principal de esa fila.
#
# Ejemplo (AJUSTA a tus nombres reales):
DEFINICION_TABLAS = {
    "maestro_postes": {
        "columnas_id_candidatas": ["codigo_poste", "id_poste"],
        "columnas_utiles": ["distrito", "red_electrica", "tipo_poste",
                             "latitud", "longitud", "fecha_instalacion"],
    },
    "lavados": {
        "columnas_id_candidatas": ["codigo_poste", "poste"],
        "columnas_utiles": ["fecha_lavado", "grupo_lavador"],
    },
    "chequeos": {
        "columnas_id_candidatas": ["codigo_poste", "id_poste_chequeo"],
        "columnas_utiles": ["fecha_chequeo", "nivel_suciedad"],
    },
    "cortos": {
        "columnas_id_candidatas": ["codigo_poste", "poste_afectado"],
        "columnas_utiles": ["fecha_corto", "nivel_corto"],  # nivel_corto: bajo/critico
    },
}


def normalizar_id_principal(df: pd.DataFrame, columnas_candidatas: List[str]) -> pd.DataFrame:
    """
    Dado un DataFrame y una lista de columnas que podrian tener el ID,
    normaliza TODAS las que existan en el df y construye una columna
    final 'id_poste_final' tomando la primera version normalizada no
    nula, en el orden dado por columnas_candidatas.

    Tambien construye 'id_poste_origen_columna' indicando de que columna
    salio finalmente el ID, util para auditoria.
    """
    df = df.copy()
    columnas_existentes = [c for c in columnas_candidatas if c in df.columns]

    if not columnas_existentes:
        raise ValueError(
            f"Ninguna de las columnas candidatas {columnas_candidatas} "
            f"existe en este DataFrame. Columnas disponibles: {list(df.columns)}"
        )

    for col in columnas_existentes:
        df = limpiar_columna_id(df, col)

    columnas_norm = [f"{c}_norm" for c in columnas_existentes]

    df["id_poste_final"] = df[columnas_norm].bfill(axis=1).iloc[:, 0]

    # Registrar de que columna original vino el ID finalmente usado
    def _origen(fila):
        for col, col_norm in zip(columnas_existentes, columnas_norm):
            if pd.notna(fila[col_norm]):
                return col
        return None

    df["id_poste_origen_columna"] = df.apply(_origen, axis=1)

    return df


# ---------------------------------------------------------------------------
# 2. RESOLUCION DE DUPLICADOS / VARIANTES DEL MISMO POSTE
# ---------------------------------------------------------------------------

def construir_tabla_canonica(maestro_normalizado: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del maestro de postes ya normalizado, construye una tabla
    canonica id_poste_final -> id_canonico (1 a 1).

    En el caso simple (sin variantes duplicadas detectadas), id_canonico
    es igual a id_poste_final. Si en el futuro detectas que un mismo
    poste fisico tiene 2 IDs distintos que coexisten en el maestro,
    agrega aqui la logica de mapeo manual o por cercania geografica
    (ver nota al final del archivo).
    """
    canonica = (
        maestro_normalizado[["id_poste_final"]]
        .drop_duplicates()
        .dropna()
        .rename(columns={"id_poste_final": "id_canonico"})
    )
    canonica["id_poste_final"] = canonica["id_canonico"]
    return canonica


def aplicar_id_canonico(df: pd.DataFrame, tabla_canonica: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza un dataframe (ya con columna id_poste_final) contra la tabla
    canonica para asignarle 'id_canonico'. Filas sin match quedan con
    id_canonico = NaN (puedes auditarlas despues con
    reporte_filas_sin_match).
    """
    return df.merge(
        tabla_canonica[["id_poste_final", "id_canonico"]],
        on="id_poste_final",
        how="left",
    )


def reporte_filas_sin_match(df: pd.DataFrame, nombre_tabla: str) -> pd.DataFrame:
    """
    Muestra cuantas filas de una tabla (lavados, chequeos, cortos) NO
    encontraron match en el maestro de postes tras normalizar IDs.
    Esto es CRITICO de revisar: si el % es alto, el problema esta en
    los patrones de PATRONES_ID (modulo 1) o en que el maestro de
    postes esta incompleto.
    """
    sin_match = df[df["id_canonico"].isna()]
    pct = len(sin_match) / len(df) * 100 if len(df) > 0 else 0
    print(f"\n[{nombre_tabla}] Filas sin match en maestro: "
          f"{len(sin_match)} de {len(df)} ({pct:.2f}%)")
    if len(sin_match) > 0:
        print("Ejemplos de IDs sin match (revisar formato):")
        print(sin_match["id_poste_final"].dropna().unique()[:15])
    return sin_match


# ---------------------------------------------------------------------------
# 3. PIPELINE COMPLETO DE CONSOLIDACION
# ---------------------------------------------------------------------------

def consolidar_todas_las_tablas(
    dataframes_crudos: Dict[str, pd.DataFrame],
    definicion: Dict[str, Dict] = DEFINICION_TABLAS,
) -> Dict[str, pd.DataFrame]:
    """
    Funcion principal de este modulo.

    Parametros:
    -----------
    dataframes_crudos: dict con la forma {"maestro_postes": df1, "lavados": df2, ...}
                       (las llaves deben coincidir con las de `definicion`)
    definicion: configuracion de columnas candidatas por tabla (ver arriba)

    Devuelve:
    ---------
    dict con los mismos dataframes, cada uno con columna 'id_canonico'
    agregada y lista para cruzar entre si.

    USO TIPICO EN TU NOTEBOOK:
        dataframes = {
            "maestro_postes": df_maestro,
            "lavados": df_lavados,
            "chequeos": df_chequeos,
            "cortos": df_cortos,
        }
        resultado = consolidar_todas_las_tablas(dataframes)
        df_lavados_listo = resultado["lavados"]
    """
    normalizados = {}
    for nombre_tabla, df in dataframes_crudos.items():
        if nombre_tabla not in definicion:
            print(f"AVISO: '{nombre_tabla}' no esta en DEFINICION_TABLAS, se omite.")
            continue
        cols_candidatas = definicion[nombre_tabla]["columnas_id_candidatas"]
        normalizados[nombre_tabla] = normalizar_id_principal(df, cols_candidatas)
        print(f"\n=== {nombre_tabla} ===")
        reporte_calidad_ids(
            normalizados[nombre_tabla],
            "id_poste_final",
            f"{normalizados[nombre_tabla]['id_poste_origen_columna'].mode().iloc[0]}_tipo_detectado"
            if normalizados[nombre_tabla]["id_poste_origen_columna"].notna().any()
            else "id_poste_origen_columna",
        )

    if "maestro_postes" not in normalizados:
        raise ValueError(
            "Se requiere una tabla 'maestro_postes' para construir el "
            "catalogo canonico de postes. Revisa DEFINICION_TABLAS."
        )

    tabla_canonica = construir_tabla_canonica(normalizados["maestro_postes"])

    resultado = {}
    for nombre_tabla, df_norm in normalizados.items():
        df_con_canonico = aplicar_id_canonico(df_norm, tabla_canonica)
        reporte_filas_sin_match(df_con_canonico, nombre_tabla)
        resultado[nombre_tabla] = df_con_canonico

    return resultado


# ---------------------------------------------------------------------------
# NOTA SOBRE EL CASO "MISMO POSTE, DOS IDS DISTINTOS COEXISTIENDO"
# ---------------------------------------------------------------------------
# Si detectas (via reporte_filas_sin_match) que hay IDs que nunca matchean
# pero corresponden a un poste real que SI esta en el maestro con otro
# codigo, la forma mas confiable de resolverlo NO es texto, es geografia:
#
#   1. Si tienes lat/long en la tabla de chequeos/lavados aunque sea
#      aproximada, puedes hacer un join espacial: para cada ID sin
#      match, buscar el poste del maestro mas cercano (< X metros) y
#      asumir que es el mismo. Usa scipy.spatial.cKDTree para esto
#      eficientemente con 40k+ postes.
#   2. Alternativamente, si el distrito + tipo de poste + un fragmento
#      del codigo coinciden, tambien es buena senal de que es el mismo.
#
# Esto se deja como mejora de la Fase 2 (no bloquea el modelo inicial),
# pero si el % de "sin match" del reporte es alto (>5-10%), vale la pena
# resolverlo antes de entrenar, porque perderias historial real de
# lavados/chequeos de postes que si existen.


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EJEMPLO SINTETICO de extremo a extremo. Reemplaza por tus dataframes
    # reales (los que ya cargaste con pandas) respetando los nombres de
    # columnas que configures en DEFINICION_TABLAS.
    # ------------------------------------------------------------------
    df_maestro = pd.DataFrame({
        "codigo_poste": ["12345A", "123456789", "54321B"],
        "distrito": ["Surco", "Miraflores", "Surco"],
        "red_electrica": ["RED-01", "RED-02", "RED-01"],
        "tipo_poste": ["concreto", "metal", "concreto"],
        "latitud": [-12.15, -12.12, -12.16],
        "longitud": [-77.00, -77.03, -77.01],
    })

    df_lavados = pd.DataFrame({
        "codigo_poste": ["12345a", "12345-A", "O23456789"],  # variantes sucias
        "fecha_lavado": pd.to_datetime(["2024-01-10", "2024-06-15", "2024-03-01"]),
        "grupo_lavador": ["G1", "G1", "G2"],
    })

    df_chequeos = pd.DataFrame({
        "codigo_poste": ["12345A", "54321B", "123456789"],
        "fecha_chequeo": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03"]),
        "nivel_suciedad": ["alto", "bajo", "medio"],
    })

    df_cortos = pd.DataFrame({
        "codigo_poste": ["12345A", "999999999"],  # el segundo no existe en maestro
        "fecha_corto": pd.to_datetime(["2024-04-20", "2024-04-21"]),
        "nivel_corto": ["critico", "bajo"],
    })

    dataframes = {
        "maestro_postes": df_maestro,
        "lavados": df_lavados,
        "chequeos": df_chequeos,
        "cortos": df_cortos,
    }

    resultado = consolidar_todas_las_tablas(dataframes)

    print("\n\n=== RESULTADO FINAL: lavados con id_canonico ===")
    print(resultado["lavados"][["codigo_poste", "id_canonico", "fecha_lavado"]])
