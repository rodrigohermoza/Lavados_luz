"""
MODULO 3: Construccion del TARGET (variable a predecir)
===========================================================

EL PROBLEMA DE FONDO:
No tienes una "falla" limpia como evento. Lo que tienes es:
  - Chequeos: cuan sucio se ve un poste en una fecha dada (proxy directo)
  - Lavados: cuando se lavo un poste (evento de "reset" de suciedad)
  - Cortos: eventos de corto bajo/critico, que PUEDEN estar relacionados
    a suciedad pero no siempre (es una causa posible, no confirmada)

LA IDEA CENTRAL:
En vez de inventar un "score de prioridad 1-10" a mano (eso seria
subjetivo y dificil de defender ante tu jefe), lo construimos en 2 pasos
medibles y luego SI lo convertimos a una escala 1-10 interpretable:

  PASO A: Para cada poste, estimar su "velocidad de ensuciamiento":
          cuantos dias tarda en pasar de limpio (recien lavado) a sucio
          (segun el chequeo). Esto se calcula directamente de tus datos
          historicos: dias_desde_ultimo_lavado en el momento de cada
          chequeo, correlacionado con el nivel de suciedad reportado.

  PASO B: Ponderar esa velocidad con el riesgo de corto. Si un poste
          tiene historial de cortos (sobre todo criticos) Y se ensucia
          rapido, su prioridad de lavado debe subir mas que un poste
          que se ensucia rapido pero nunca ha tenido cortos.

  PASO C: Convertir el resultado combinado en un score 1-10 por
          PERCENTILES (no por umbrales arbitrarios), de forma que el
          score sea relativo a la distribucion real de tus 40k postes,
          y directamente util para priorizar bajo restriccion de
          capacidad (los lavadores solo pueden hacer ~120/dia).

Este modulo asume que ya corriste el modulo 2 y tienes las tablas
(maestro, lavados, chequeos, cortos) con columna 'id_canonico' comun.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# PASO A: VELOCIDAD DE ENSUCIAMIENTO POR POSTE
# ---------------------------------------------------------------------------

def construir_eventos_chequeo_con_antiguedad_lavado(
    chequeos: pd.DataFrame,
    lavados: pd.DataFrame,
    maestro_fechas_instalacion: pd.DataFrame = None,
    col_id: str = "id_canonico",
    col_fecha_chequeo: str = "fecha_chequeo",
    col_fecha_lavado: str = "fecha_lavado",
    col_nivel_suciedad: str = "nivel_suciedad",
    col_fecha_instalacion: str = "fecha_instalacion",
) -> pd.DataFrame:
    """
    Para cada chequeo, calcula cuantos dias habian pasado desde el
    ULTIMO lavado anterior a esa fecha de chequeo (para ese mismo poste).

    Esto es la pieza clave: nos da pares (dias_desde_lavado, nivel_suciedad)
    que son la base empirica de la velocidad de ensuciamiento.

    IMPORTANTE - postes nunca lavados: si un poste no tiene NINGUN lavado
    registrado antes de la fecha de chequeo, usamos su fecha de
    instalacion (si se provee `maestro_fechas_instalacion`) como
    referencia de "ultimo evento de limpieza conocido". Sin este
    fallback, estos postes quedarian SIN pendiente calculable y se
    perderian del ranking final -- justo el tipo de poste que mas
    interesa detectar (sucio, con cortos, nunca lavado). Si tampoco hay
    fecha de instalacion disponible, se deja NaN y se documenta en el
    reporte de cobertura (ver modulo 4).

    Devuelve el dataframe de chequeos con una columna nueva:
    'dias_desde_ultimo_lavado' y 'origen_referencia' (lavado_previo /
    fecha_instalacion / sin_referencia) para que puedas auditar de
    donde sale cada numero.
    """
    chequeos = chequeos.copy().sort_values([col_id, col_fecha_chequeo])
    lavados = lavados.copy().sort_values([col_id, col_fecha_lavado])

    fechas_instalacion_por_poste = {}
    if maestro_fechas_instalacion is not None and col_fecha_instalacion in maestro_fechas_instalacion.columns:
        fechas_instalacion_por_poste = (
            maestro_fechas_instalacion.dropna(subset=[col_fecha_instalacion])
            .set_index(col_id)[col_fecha_instalacion]
            .to_dict()
        )

    resultados = []
    # Agrupar lavados por poste para busqueda eficiente
    lavados_por_poste = {
        pid: grupo[col_fecha_lavado].sort_values().to_numpy()
        for pid, grupo in lavados.groupby(col_id)
    }

    for _, fila in chequeos.iterrows():
        pid = fila[col_id]
        fecha_chk = pd.Timestamp(fila[col_fecha_chequeo])
        fechas_lavado_poste = lavados_por_poste.get(pid, np.array([], dtype="datetime64[ns]"))

        # Buscar el ultimo lavado ANTERIOR (o igual) a la fecha de chequeo
        fechas_validas = fechas_lavado_poste[fechas_lavado_poste <= np.datetime64(fecha_chk)]

        if len(fechas_validas) > 0:
            ultimo_lavado = pd.Timestamp(fechas_validas.max())
            dias = (fecha_chk - ultimo_lavado).days
            origen = "lavado_previo"
        elif pid in fechas_instalacion_por_poste:
            fecha_instalacion = pd.Timestamp(fechas_instalacion_por_poste[pid])
            ultimo_lavado = fecha_instalacion
            dias = (fecha_chk - fecha_instalacion).days
            origen = "fecha_instalacion"
        else:
            ultimo_lavado = pd.NaT
            dias = np.nan  # no hay ninguna referencia disponible
            origen = "sin_referencia"

        resultados.append({
            col_id: pid,
            col_fecha_chequeo: fecha_chk,
            col_nivel_suciedad: fila.get(col_nivel_suciedad),
            "fecha_ultimo_lavado_previo": ultimo_lavado,
            "dias_desde_ultimo_lavado": dias,
            "origen_referencia": origen,
        })

    return pd.DataFrame(resultados)


def mapear_nivel_suciedad_a_score(nivel_suciedad: pd.Series) -> pd.Series:
    """
    Convierte el nivel de suciedad categorico (texto) a un score numerico
    0-1 para poder hacer regresion. AJUSTA este mapeo a las categorias
    REALES que uses en tus chequeos (puede que tengas mas niveles, o
    nombres distintos como 'leve'/'moderado'/'severo').
    """
    mapeo = {
        "limpio": 0.0,
        "bajo": 0.25,
        "leve": 0.25,
        "medio": 0.5,
        "moderado": 0.5,
        "alto": 0.85,
        "severo": 0.85,
        "critico": 1.0,
        "muy_sucio": 1.0,
    }
    normalizado = nivel_suciedad.astype(str).str.strip().str.lower()
    return normalizado.map(mapeo)


def estimar_velocidad_ensuciamiento_por_poste(
    eventos_chequeo: pd.DataFrame,
    col_id: str = "id_canonico",
) -> pd.DataFrame:
    """
    Para cada poste, ajusta una relacion simple entre 'dias_desde_ultimo_lavado'
    y 'score_suciedad' usando regresion lineal por poste (si tiene >= 2
    chequeos utiles) para estimar la PENDIENTE: cuanto sube el score de
    suciedad por cada dia que pasa sin lavar.

    Si un poste tiene menos de 2 chequeos utiles, no se puede estimar su
    propia pendiente -> se deja NaN y luego se imputa con el promedio de
    su grupo (mismo tipo de poste / mismo distrito) en el modulo 4.

    Devuelve un dataframe con: id_canonico, pendiente_ensuciamiento_dia,
    n_chequeos_usados, score_suciedad_promedio, score_suciedad_max.
    """
    eventos_chequeo = eventos_chequeo.copy()
    eventos_chequeo["score_suciedad"] = mapear_nivel_suciedad_a_score(
        eventos_chequeo["nivel_suciedad"]
    )

    utiles = eventos_chequeo.dropna(subset=["dias_desde_ultimo_lavado", "score_suciedad"])

    resultados = []
    for pid, grupo in utiles.groupby(col_id):
        grupo = grupo.sort_values("dias_desde_ultimo_lavado")
        n = len(grupo)

        if n >= 2 and grupo["dias_desde_ultimo_lavado"].nunique() > 1:
            # Regresion lineal simple (1 variable): pendiente = cov(x,y)/var(x)
            x = grupo["dias_desde_ultimo_lavado"].to_numpy(dtype=float)
            y = grupo["score_suciedad"].to_numpy(dtype=float)
            pendiente = np.polyfit(x, y, 1)[0]
            pendiente = max(pendiente, 0.0)  # forzar no-negativo (suciedad no "mejora" sola)
        else:
            pendiente = np.nan  # se imputara despues con el promedio del grupo

        resultados.append({
            col_id: pid,
            "pendiente_ensuciamiento_dia": pendiente,
            "n_chequeos_usados": n,
            "score_suciedad_promedio": grupo["score_suciedad"].mean(),
            "score_suciedad_max": grupo["score_suciedad"].max(),
            "dias_desde_lavado_en_ultimo_chequeo": grupo["dias_desde_ultimo_lavado"].iloc[-1],
        })

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# PASO B: SEÑAL DE RIESGO DE CORTO ASOCIADO A SUCIEDAD
# ---------------------------------------------------------------------------

def construir_senal_riesgo_corto(
    cortos: pd.DataFrame,
    col_id: str = "id_canonico",
    col_fecha_corto: str = "fecha_corto",
    col_nivel_corto: str = "nivel_corto",
    ventana_relevancia_dias: int = 365,
) -> pd.DataFrame:
    """
    Construye, por poste, un indicador de riesgo basado en su historial
    de cortos. Pondera mas los cortos CRITICOS y los mas RECIENTES.

    No afirma causalidad (sabemos que el corto puede o no deberse a
    suciedad), pero es razonable usarlo como FACTOR DE PONDERACION del
    riesgo: si un poste con historial de cortos criticos ademas se
    ensucia rapido, prevenir tiene mas valor que en un poste sin
    historial de cortos.

    Devuelve: id_canonico, n_cortos_bajos, n_cortos_criticos,
    score_riesgo_corto (0 a 1, normalizado).
    """
    cortos = cortos.copy()
    peso_nivel = {"bajo": 1.0, "critico": 3.0}  # AJUSTA si tienes mas niveles
    cortos["peso_nivel"] = (
        cortos[col_nivel_corto].astype(str).str.strip().str.lower().map(peso_nivel).fillna(1.0)
    )

    hoy = pd.Timestamp.now()
    dias_desde_corto = (hoy - cortos[col_fecha_corto]).dt.days.clip(lower=0)
    # Peso de recencia: decae exponencialmente, los cortos de hace mas de
    # ~1 anio pesan poco pero no cero (vida media configurable)
    cortos["peso_recencia"] = np.exp(-dias_desde_corto / ventana_relevancia_dias)

    cortos["peso_total"] = cortos["peso_nivel"] * cortos["peso_recencia"]

    agregado = cortos.groupby(col_id).agg(
        n_cortos_bajos=(col_nivel_corto, lambda s: (s.str.lower() == "bajo").sum()),
        n_cortos_criticos=(col_nivel_corto, lambda s: (s.str.lower() == "critico").sum()),
        score_riesgo_corto_crudo=("peso_total", "sum"),
    ).reset_index()

    # Normalizar a 0-1 usando percentil (robusto a outliers, mejor que min-max)
    if agregado["score_riesgo_corto_crudo"].max() > 0:
        agregado["score_riesgo_corto"] = (
            agregado["score_riesgo_corto_crudo"].rank(pct=True)
        )
    else:
        agregado["score_riesgo_corto"] = 0.0

    return agregado.drop(columns=["score_riesgo_corto_crudo"])


# ---------------------------------------------------------------------------
# PASO C: TARGET FINAL COMBINADO (escala 1-10)
# ---------------------------------------------------------------------------

def construir_target_prioridad(
    velocidad_ensuciamiento: pd.DataFrame,
    riesgo_corto: pd.DataFrame,
    col_id: str = "id_canonico",
    peso_velocidad: float = 0.7,
    peso_riesgo_corto: float = 0.3,
) -> pd.DataFrame:
    """
    Combina la velocidad de ensuciamiento (señal principal, mas confiable
    porque viene directo de chequeos) con el riesgo de corto (señal
    secundaria, de ponderacion) en un score final 0-1, y lo convierte a
    una escala 1-10 por PERCENTILES sobre el universo de postes.

    AJUSTA peso_velocidad / peso_riesgo_corto segun lo que tu jefe
    priorice: si el negocio le da mas peso a evitar cortos que a
    suciedad pura, sube peso_riesgo_corto.

    Devuelve: id_canonico, score_prioridad_0_1, prioridad_1_10,
    y las columnas intermedias para poder explicar el score (importante
    para que tu jefe confie en el ranking).
    """
    combinado = velocidad_ensuciamiento.merge(riesgo_corto, on=col_id, how="left")

    combinado["score_riesgo_corto"] = combinado["score_riesgo_corto"].fillna(0.0)
    combinado["n_cortos_bajos"] = combinado["n_cortos_bajos"].fillna(0)
    combinado["n_cortos_criticos"] = combinado["n_cortos_criticos"].fillna(0)

    # Normalizar la pendiente de ensuciamiento a 0-1 por percentil
    # (la pendiente NaN, de postes con <2 chequeos, se imputa en modulo 4
    # con el promedio del grupo; aqui solo normalizamos lo disponible)
    combinado["score_velocidad"] = combinado["pendiente_ensuciamiento_dia"].rank(pct=True)

    combinado["score_prioridad_0_1"] = (
        peso_velocidad * combinado["score_velocidad"].fillna(0)
        + peso_riesgo_corto * combinado["score_riesgo_corto"]
    )

    # Conversion a escala 1-10 por percentil (decil), interpretable y
    # directamente usable para priorizar bajo restriccion de capacidad:
    # el decil 10 son el 10% de postes mas urgentes de lavar.
    combinado["prioridad_1_10"] = (
        pd.qcut(combinado["score_prioridad_0_1"], 10, labels=False, duplicates="drop") + 1
    )

    return combinado


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EJEMPLO SINTETICO. Sustituye por tus tablas reales (ya con
    # id_canonico asignado por el modulo 2).
    # ------------------------------------------------------------------
    chequeos = pd.DataFrame({
        "id_canonico": ["12345A", "12345A", "54321B", "54321B", "99999C"],
        "fecha_chequeo": pd.to_datetime([
            "2024-02-01", "2024-04-01", "2024-02-15", "2024-05-15", "2024-03-01"
        ]),
        "nivel_suciedad": ["medio", "alto", "bajo", "medio", "critico"],
    })

    lavados = pd.DataFrame({
        "id_canonico": ["12345A", "54321B"],
        "fecha_lavado": pd.to_datetime(["2024-01-01", "2024-01-15"]),
    })

    # 99999C nunca fue lavado (no aparece en `lavados`), pero SI tiene
    # fecha de instalacion en el maestro -> se usa como referencia para
    # no perderlo del calculo (ver docstring de la funcion).
    maestro_fechas = pd.DataFrame({
        "id_canonico": ["99999C"],
        "fecha_instalacion": pd.to_datetime(["2023-11-01"]),
    })

    cortos = pd.DataFrame({
        "id_canonico": ["12345A", "99999C", "99999C"],
        "fecha_corto": pd.to_datetime(["2024-03-10", "2024-01-05", "2024-04-01"]),
        "nivel_corto": ["bajo", "critico", "critico"],
    })

    eventos = construir_eventos_chequeo_con_antiguedad_lavado(
        chequeos, lavados, maestro_fechas_instalacion=maestro_fechas
    )
    print("=== Eventos de chequeo con antiguedad de lavado ===")
    print(eventos[["id_canonico", "dias_desde_ultimo_lavado", "origen_referencia"]])

    velocidad = estimar_velocidad_ensuciamiento_por_poste(eventos)
    print("\n=== Velocidad de ensuciamiento por poste ===")
    print(velocidad)

    riesgo_corto = construir_senal_riesgo_corto(cortos)
    print("\n=== Riesgo de corto por poste ===")
    print(riesgo_corto)

    target = construir_target_prioridad(velocidad, riesgo_corto)
    print("\n=== Target final de prioridad ===")
    print(target[["id_canonico", "score_prioridad_0_1", "prioridad_1_10",
                   "pendiente_ensuciamiento_dia", "score_riesgo_corto"]])
