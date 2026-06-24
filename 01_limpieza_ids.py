"""
MODULO 1: Limpieza y normalizacion de identificadores de postes
==================================================================

PROBLEMA QUE RESUELVE:
- Los IDs de poste aparecen en varias columnas (a veces distintos nombres
  segun el archivo: 'codigo_poste', 'cod_poste', 'id_poste', 'numero', etc.)
- Tienen errores de tipeo / OCR: 'O' en vez de '0', espacios, guiones,
  mayusculas/minusculas inconsistentes.
- Distintos tipos de poste tienen distintos formatos de codigo:
  5 numeros + letra (ej: 12345A), 9 numeros (ej: 123456789), etc.

ESTRATEGIA:
1. Definir una funcion de normalizacion de texto (limpieza generica).
2. Definir patrones (regex) para los formatos de codigo conocidos.
3. Clasificar cada ID segun el patron que matchea -> columna 'tipo_id_detectado'.
4. Generar una columna 'id_poste_normalizado' unica y confiable para usar
   como llave de cruce entre TODAS las tablas.

IMPORTANTE: Ajusta la lista PATRONES_ID con los formatos reales que tengas.
Lo que ves aqui son los 2 que mencionaste (5 numeros+letra, 9 numeros) mas
algunos genericos de respaldo. Anade los que falten.
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


# ---------------------------------------------------------------------------
# 1. NORMALIZACION DE TEXTO (errores de tipeo / OCR)
# ---------------------------------------------------------------------------

# Mapeo de caracteres que se confunden tipicamente en captura manual u OCR.
# Ojo: este mapeo es agresivo (asume que un poste no deberia tener letras tipo
# O, I, S, B salvo que sean parte real del codigo). Revisa si tus codigos
# reales SI usan letras como parte significativa antes de aplicar todo el
# mapeo a la cadena completa. Por eso separamos "limpieza de simbolos" de
# "correccion de confusiones letra/numero", y esta ultima se aplica
# selectivamente segun el patron esperado (ver mas abajo).
MAPA_CONFUSIONES_OCR = {
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "L": "1",  # comentar esta linea si "L" es una letra valida en tus codigos
    "S": "5",  # comentar si "S" es valida (algunos formatos la usan como letra real)
    "B": "8",  # comentar si "B" es valida
}


def normalizar_texto_basico(valor) -> Optional[str]:
    """
    Limpieza generica de un identificador en bruto.
    - Convierte a string
    - Quita espacios al inicio/fin y espacios internos
    - Quita guiones, puntos, guion bajo, slash
    - Pasa todo a mayusculas (estandar para comparar)
    - Devuelve None si el valor es nulo o vacio tras limpiar
    """
    if pd.isna(valor):
        return None

    texto = str(valor).strip()

    if texto == "" or texto.lower() in ("nan", "none", "null", "n/a", "-"):
        return None

    # Si pandas leyo un numero como float (ej 12345.0), quitar el ".0"
    texto = re.sub(r"\.0$", "", texto)

    # Quitar separadores comunes: espacios, guiones, puntos, guion bajo, slash
    texto = re.sub(r"[\s\-\._/]", "", texto)

    texto = texto.upper()

    return texto if texto != "" else None


def corregir_confusiones_ocr(texto: str, esperar_solo_digitos: bool = False) -> str:
    """
    Aplica correccion de caracteres confundibles SOLO cuando tiene sentido.

    Si esperar_solo_digitos=True, asumimos que el codigo deberia ser
    puramente numerico (ej: el formato de 9 numeros) y por lo tanto
    cualquier letra remanente es casi seguro un error de tipeo -> se
    corrige con el mapa OCR.

    Si esperar_solo_digitos=False, NO tocamos las letras (porque el
    formato esperado mezcla numeros y letra real, ej: 12345A) salvo
    que sea un caracter claramente ambiguo dentro de la PARTE NUMERICA.
    """
    if texto is None:
        return None

    if esperar_solo_digitos:
        return "".join(MAPA_CONFUSIONES_OCR.get(ch, ch) for ch in texto)

    return texto


# ---------------------------------------------------------------------------
# 2. PATRONES DE FORMATO DE CODIGO CONOCIDOS
# ---------------------------------------------------------------------------
# AJUSTA ESTA LISTA con los formatos reales de tu empresa.
# El orden importa: se evalua de arriba hacia abajo y se usa el PRIMER match.
# Pon los patrones mas especificos primero.

PATRONES_ESPECIFICOS = [
    # (nombre_tipo, regex, solo_digitos_esperado)
    # Estos son los formatos REALES y conocidos de poste. Anade aqui los
    # que falten en tu empresa. Se evaluan primero, antes de cualquier
    # patron generico, para minimizar falsos "no reconocidos".
    ("5DIGITOS_1LETRA", r"^\d{5}[A-Z]$", False),
    ("9DIGITOS", r"^\d{9}$", True),
    ("1LETRA_5DIGITOS", r"^[A-Z]\d{5}$", False),
]

PATRONES_GENERICOS = [
    # Patrones de respaldo: solo se usan si NINGUN patron especifico
    # matcheo, ni siquiera tras intentar corregir errores OCR. Por eso
    # van separados: si los mezclamos con los especificos, un ID con
    # error OCR (ej. "O2345A") podria matchear el generico ANTES de que
    # probemos corregirlo, y nos quedariamos con la version sucia.
    ("GENERICO_NUMERICO", r"^\d{6,12}$", True),
    ("GENERICO_ALFANUM", r"^[A-Z0-9]{4,12}$", False),
]

# Lista combinada, usada solo por reporte_calidad_ids / utilidades que
# necesiten iterar sobre "todos" los patrones conocidos.
PATRONES_ID = PATRONES_ESPECIFICOS + PATRONES_GENERICOS


def detectar_tipo_y_normalizar(valor_crudo) -> pd.Series:
    """
    Toma un valor crudo de ID y devuelve:
    - id_normalizado: el ID limpio y corregido (o None si no es valido)
    - tipo_id_detectado: el patron que matcheo (o 'NO_RECONOCIDO' / 'NULO')

    Orden de intentos (importante para no perder informacion):
    1. Patrones especificos tal cual viene el texto limpio.
    2. Patrones especificos tras corregir errores OCR (numerico puro).
    3. Patrones especificos mixtos (digitos+letra) corrigiendo OCR
       solo en la parte que se espera numerica.
    4. Patrones genericos de respaldo (sin correccion), como ultimo recurso.
    5. Si nada matchea, se marca NO_RECONOCIDO pero se conserva el valor
       limpio para que puedas auditarlo, en vez de descartarlo.

    Esta funcion se aplica fila por fila y columna por columna durante
    la consolidacion (ver modulo 2).
    """
    texto = normalizar_texto_basico(valor_crudo)

    if texto is None:
        return pd.Series({"id_normalizado": None, "tipo_id_detectado": "NULO"})

    # PASO 1: patrones especificos tal cual
    for nombre_tipo, patron, solo_digitos in PATRONES_ESPECIFICOS:
        if re.match(patron, texto):
            return pd.Series({
                "id_normalizado": texto,
                "tipo_id_detectado": nombre_tipo,
            })

    # PASO 2: patrones especificos puramente numericos, tras corregir OCR
    texto_corregido = corregir_confusiones_ocr(texto, esperar_solo_digitos=True)
    for nombre_tipo, patron, solo_digitos in PATRONES_ESPECIFICOS:
        if solo_digitos and re.match(patron, texto_corregido):
            return pd.Series({
                "id_normalizado": texto_corregido,
                "tipo_id_detectado": f"{nombre_tipo}_CORREGIDO_OCR",
            })

    # PASO 3: patrones especificos mixtos (digitos + 1 letra), corrigiendo
    # OCR solo en la parte que se espera numerica (ej. "O2345A" -> "02345A")
    LONGITUDES_ESPERADAS = {
        "5DIGITOS_1LETRA": 6,
        "1LETRA_5DIGITOS": 6,
    }
    for nombre_tipo, patron, solo_digitos in PATRONES_ESPECIFICOS:
        if nombre_tipo not in LONGITUDES_ESPERADAS:
            continue
        if len(texto) != LONGITUDES_ESPERADAS[nombre_tipo]:
            continue
        if nombre_tipo == "5DIGITOS_1LETRA":
            parte_num, parte_letra = texto[:5], texto[5:]
            parte_num_corregida = corregir_confusiones_ocr(parte_num, esperar_solo_digitos=True)
            candidato = parte_num_corregida + parte_letra
        else:  # 1LETRA_5DIGITOS
            parte_letra, parte_num = texto[0], texto[1:]
            parte_num_corregida = corregir_confusiones_ocr(parte_num, esperar_solo_digitos=True)
            candidato = parte_letra + parte_num_corregida
        if re.match(patron, candidato):
            return pd.Series({
                "id_normalizado": candidato,
                "tipo_id_detectado": f"{nombre_tipo}_CORREGIDO_OCR",
            })

    # PASO 4: patrones genericos de respaldo, sin correccion OCR (si el
    # ID ya es "razonable" aunque no calce con ningun formato conocido,
    # mejor conservarlo reconocido como generico que perderlo)
    for nombre_tipo, patron, solo_digitos in PATRONES_GENERICOS:
        if re.match(patron, texto):
            return pd.Series({
                "id_normalizado": texto,
                "tipo_id_detectado": nombre_tipo,
            })

    # PASO 5: no reconocido por ningun patron. Se conserva limpio y
    # marcado para auditoria manual posterior (ver reporte_calidad_ids).
    return pd.Series({
        "id_normalizado": texto,
        "tipo_id_detectado": "NO_RECONOCIDO",
    })


# ---------------------------------------------------------------------------
# 3. FUNCION PRINCIPAL: limpiar un DataFrame dado, detectando la(s)
#    columna(s) candidatas a ser identificador de poste
# ---------------------------------------------------------------------------

def limpiar_columna_id(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Aplica la deteccion/normalizacion a una columna especifica de un
    DataFrame y agrega dos columnas nuevas:
    - {columna}_norm
    - {columna}_tipo_detectado

    Uso:
        df = limpiar_columna_id(df, "codigo_poste")
    """
    resultado = df[columna].apply(detectar_tipo_y_normalizar)
    df[f"{columna}_norm"] = resultado["id_normalizado"]
    df[f"{columna}_tipo_detectado"] = resultado["tipo_id_detectado"]
    return df


def reporte_calidad_ids(df: pd.DataFrame, columna_norm: str, columna_tipo: str) -> pd.DataFrame:
    """
    Genera un resumen rapido de cuantos IDs cayeron en cada categoria.
    Util para correr DESPUES de limpiar_columna_id y revisar que tan
    sucios estan tus datos reales antes de seguir.
    """
    resumen = df[columna_tipo].value_counts(dropna=False).reset_index()
    resumen.columns = ["tipo_id_detectado", "conteo"]
    resumen["porcentaje"] = (resumen["conteo"] / len(df) * 100).round(2)

    n_nulos = df[columna_norm].isna().sum()
    n_unicos = df[columna_norm].nunique()

    print(f"\n--- Reporte de calidad para columna normalizada '{columna_norm}' ---")
    print(f"Total filas: {len(df)}")
    print(f"IDs nulos/vacios: {n_nulos} ({n_nulos/len(df)*100:.2f}%)")
    print(f"IDs unicos detectados: {n_unicos}")
    print(resumen.to_string(index=False))

    return resumen


# ---------------------------------------------------------------------------
# EJEMPLO DE USO (ajusta nombres de archivo/columna a tu caso real)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo sintetico para que veas el comportamiento ANTES de correrlo
    # contra tus datos reales. Borra este bloque y reemplaza por tu carga real.
    ejemplo = pd.DataFrame({
        "codigo_poste": [
            "12345A", "12345a", " 12345-A", "O2345A",   # variantes del mismo poste
            "123456789", "12345678O",                    # 9 digitos, uno con error OCR
            "B2345A",                                      # otro caso con confusion letra
            None, "", "n/a",
        ]
    })

    ejemplo = limpiar_columna_id(ejemplo, "codigo_poste")
    print(ejemplo)
    reporte_calidad_ids(ejemplo, "codigo_poste_norm", "codigo_poste_tipo_detectado")
