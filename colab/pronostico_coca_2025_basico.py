# %% [markdown]
# # Pronóstico de coca 2025 – versión básica (total nacional)
#
# Comparamos 4 modelos para pronosticar las hectáreas de coca de **2025 a nivel nacional**:
# regresión lineal, regresión polinómica, Holt y ARIMA.
#
# **¿Cómo elegimos el mejor?** Hacemos una "prueba en el pasado": cada modelo pronostica los años
# anteriores usando sólo los años previos a cada uno, y medimos cuánto se equivocó (MAE y RMSE).
# Probamos cada modelo con ventanas de **3, 5 y 10 años** de historia (las que permitan los datos).
#
# **Punto de partida:** el dataframe `fact_cultivos_raw` del notebook de limpieza.
#
# **ARIMA** tiene tres números (p, d, q). Probamos todas las combinaciones y nos quedamos con la mejor.

# %%
# ============================================================
# 1. LIBRERÍAS
# ============================================================
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels avisa mucho con series cortas

# %%
# ============================================================
# 2. PARTIR DE fact_cultivos_raw (ya cargado en el notebook de limpieza)
# ============================================================
# Ejecuta antes el notebook de limpieza para que fact_cultivos_raw exista en memoria.
# Columnas esperadas: cod_municipio | anio | hectareas_coca
print(f"fact_cultivos_raw: {len(fact_cultivos_raw):,} filas | "
      f"años {fact_cultivos_raw['anio'].min()}-{fact_cultivos_raw['anio'].max()}")

ANIO_PRONOSTICO = 2025

# Serie del TOTAL NACIONAL: suma de todos los municipios por año (sólo años anteriores a 2025)
serie = fact_cultivos_raw.groupby("anio")["hectareas_coca"].sum()
serie = serie[serie.index < ANIO_PRONOSTICO]
print(serie.round(0))

# Control de calidad: ¿algún año cae a menos de la mitad de sus vecinos? (pasa con 2020 si
# parsear_hectareas no reconoce el formato "8.832,92")
for anio in serie.index[1:-1]:
    vecinos = (serie.loc[anio - 1] + serie.loc[anio + 1]) / 2
    if serie.loc[anio] < 0.5 * vecinos:
        print(f"⚠️ {anio}: {serie.loc[anio]:,.0f} ha, muy por debajo de sus años vecinos (~{vecinos:,.0f} ha). "
              "Revisa parsear_hectareas en el notebook de limpieza.")

# %%
# ============================================================
# 3. GRÁFICO DE LA SERIE HISTÓRICA
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(serie.index, serie.values, marker="o", color="black")
plt.title("Hectáreas de coca en Colombia (total nacional)")
plt.ylabel("Hectáreas")
plt.grid(alpha=0.3)
plt.show()

# %%
# ============================================================
# 4. PARÁMETROS DE LA PRUEBA (se ajustan a los años disponibles)
# ============================================================
# Para probar una ventana de N años necesitamos N años de historia ANTES de cada año de prueba.
# Exigimos al menos 2 años de prueba por ventana; si no alcanzan los datos, la ventana se omite.
VENTANAS_DESEADAS = [3, 5, 10]
MIN_ANIOS_PRUEBA = 2
MAX_ANIOS_PRUEBA = 6                      # como máximo probamos los últimos 6 años (2019-2024)

primer_anio, ultimo_anio = serie.index.min(), serie.index.max()
VENTANAS = [v for v in VENTANAS_DESEADAS if ultimo_anio - (primer_anio + v) + 1 >= MIN_ANIOS_PRUEBA]
omitidas = [v for v in VENTANAS_DESEADAS if v not in VENTANAS]

# Todas las ventanas se evalúan en los MISMOS años para que la comparación sea justa
inicio_prueba = max(primer_anio + max(VENTANAS), ultimo_anio - MAX_ANIOS_PRUEBA + 1)
ANIOS_PRUEBA = list(range(inicio_prueba, ultimo_anio + 1))

print(f"Ventanas evaluadas: {VENTANAS}")
print(f"Años de prueba: {ANIOS_PRUEBA[0]}-{ANIOS_PRUEBA[-1]} ({len(ANIOS_PRUEBA)} años)")
if omitidas:
    print(f"⚠️ Ventanas omitidas por falta de historia: {omitidas}. fact_cultivos_raw empieza en {primer_anio}; "
          f"para usarlas, carga desde 2001 en el notebook de limpieza: range(2001, 2026).")

# %%
# ============================================================
# 5. MEJOR COMBINACIÓN DE ARIMA (p, d, q)
# ============================================================
# p = cuántos años anteriores usa para explicar el actual (parte "autorregresiva")
# d = cuántas veces se "diferencia" la serie (0 = usa los valores, 1 = usa los cambios año a año,
#     2 = usa el cambio de los cambios)
# q = cuántos errores pasados usa para corregirse (parte de "media móvil")
#
# Probamos p, d, q entre 0 y 2 (27 combinaciones) con cada ventana, y elegimos por el MAE
# de la prueba en el pasado (no sólo por el ajuste a los datos).
#
# Con pocos años no se pueden estimar muchos parámetros: si la combinación pide más parámetros
# que datos disponibles, se descarta.

def pronostico_arima(y, orden):
    p, d, q = orden
    if len(y) - d <= p + q + 1:      # muy pocos datos para esta combinación
        return None
    tendencia = {0: "c", 1: "t", 2: "n"}[d]   # constante / deriva / nada, según d
    try:
        ajuste = ARIMA(y, order=orden, trend=tendencia).fit()
        return max(0.0, float(ajuste.forecast(1)[0]))
    except Exception:
        return None


combinaciones = list(itertools.product(range(3), range(3), range(3)))   # (0,0,0) ... (2,2,2)
filas = []
for ventana in VENTANAS:
    for orden in combinaciones:
        errores = []
        for anio in ANIOS_PRUEBA:
            y = serie.loc[anio - ventana: anio - 1].values.astype(float)
            pron = pronostico_arima(y, orden)
            if pron is None:
                break
            errores.append(pron - serie.loc[anio])
        if len(errores) == len(ANIOS_PRUEBA):          # sólo combinaciones que funcionaron todos los años
            errores = np.array(errores)
            filas.append({"ventana": ventana, "orden (p,d,q)": orden,
                          "MAE": np.abs(errores).mean(), "RMSE": np.sqrt((errores ** 2).mean())})

resultados_arima = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)
print("Top 10 combinaciones ARIMA (menor MAE = mejor):")
print(resultados_arima.head(10).round(0).to_string())

# La mejor combinación para cada ventana (la usaremos en la comparación con los otros modelos)
MEJOR_ORDEN = resultados_arima.groupby("ventana").first()["orden (p,d,q)"].to_dict()
print("\nMejor orden ARIMA por ventana:", MEJOR_ORDEN)

# %%
# ============================================================
# 6. LOS CUATRO MODELOS
# ============================================================
# Cada función recibe los últimos años (y) y devuelve el pronóstico del año siguiente.

def modelo_lineal(y, ventana):
    """Recta que mejor pasa por los puntos, prolongada un año."""
    x = np.arange(len(y))
    return np.polyval(np.polyfit(x, y, 1), len(y))


def modelo_polinomico(y, ventana):
    """Curva (parábola, grado 2) prolongada un año."""
    x = np.arange(len(y))
    return np.polyval(np.polyfit(x, y, 2), len(y))


def modelo_holt(y, ventana):
    """Suavizado exponencial de Holt: sigue el nivel y la tendencia recientes."""
    try:
        return Holt(y, initialization_method="estimated").fit().forecast(1)[0]
    except Exception:
        return modelo_lineal(y, ventana)     # respaldo si la serie es demasiado corta


def modelo_arima(y, ventana):
    """ARIMA con la mejor combinación (p,d,q) encontrada para esa ventana."""
    return pronostico_arima(y, MEJOR_ORDEN[ventana])


MODELOS = {
    "Regresión lineal": modelo_lineal,
    "Regresión polinómica": modelo_polinomico,
    "Holt": modelo_holt,
    "ARIMA": modelo_arima,
}

# %%
# ============================================================
# 7. PRUEBA EN EL PASADO: 4 MODELOS x 3 VENTANAS
# ============================================================
filas = []
for ventana in VENTANAS:
    for nombre, modelo in MODELOS.items():
        for anio in ANIOS_PRUEBA:
            y = serie.loc[anio - ventana: anio - 1].values.astype(float)
            pron = max(0.0, float(modelo(y, ventana)))   # no hay hectáreas negativas
            filas.append({"modelo": nombre, "ventana": ventana, "anio": anio,
                          "pronostico": pron, "real": serie.loc[anio]})

prueba = pd.DataFrame(filas)
prueba["error"] = prueba["pronostico"] - prueba["real"]

tabla_errores = (prueba.groupby(["modelo", "ventana"])["error"]
                 .agg(MAE=lambda e: e.abs().mean(), RMSE=lambda e: np.sqrt((e ** 2).mean()))
                 .reset_index().sort_values("MAE"))
print("Error de cada modelo y ventana (en hectáreas, menor = mejor):")
print(tabla_errores.round(0).to_string(index=False))

mejor = tabla_errores.iloc[0]
MEJOR_MODELO, MEJOR_VENTANA = mejor["modelo"], int(mejor["ventana"])
print(f"\n🏆 Mejor: {MEJOR_MODELO} con {MEJOR_VENTANA} años (MAE = {mejor['MAE']:,.0f} ha)")

# %%
# ============================================================
# 8. GRÁFICO: COMPARACIÓN DE ERRORES
# ============================================================
tabla_mae = tabla_errores.pivot(index="modelo", columns="ventana", values="MAE")
tabla_mae.plot(kind="bar", figsize=(9, 4), color=["#86b6ef", "#2a78d6", "#104281"][:len(VENTANAS)], rot=0)
plt.title(f"Error promedio (MAE) al pronosticar {ANIOS_PRUEBA[0]}-{ANIOS_PRUEBA[-1]} – menor es mejor")
plt.ylabel("Hectáreas")
plt.xlabel("")
plt.legend(title="Ventana (años)")
plt.grid(axis="y", alpha=0.3)
plt.show()

# %%
# ============================================================
# 9. GRÁFICO: PRONÓSTICOS DE PRUEBA VS REAL (mejor ventana)
# ============================================================
colores = {"Regresión lineal": "#2a78d6", "Regresión polinómica": "#eb6834",
           "Holt": "#1baf7a", "ARIMA": "#eda100"}
plt.figure(figsize=(10, 4))
plt.plot(serie.loc[ultimo_anio - 12:].index, serie.loc[ultimo_anio - 12:].values, marker="o", color="black", label="Real")
for nombre in MODELOS:
    d = prueba[(prueba["modelo"] == nombre) & (prueba["ventana"] == MEJOR_VENTANA)]
    plt.plot(d["anio"], d["pronostico"], marker="o", color=colores[nombre], label=nombre)
plt.title(f"Pronósticos de prueba vs. valor real (ventana de {MEJOR_VENTANA} años)")
plt.ylabel("Hectáreas")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# ============================================================
# 10. PRONÓSTICO 2025
# ============================================================
filas = []
for ventana in VENTANAS:
    y = serie.loc[ANIO_PRONOSTICO - ventana: ANIO_PRONOSTICO - 1].values.astype(float)
    for nombre, modelo in MODELOS.items():
        filas.append({"modelo": nombre, "ventana": ventana,
                      "pronostico_2025": max(0.0, float(modelo(y, ventana)))})
pronosticos = pd.DataFrame(filas).merge(tabla_errores, on=["modelo", "ventana"])
pronosticos = pronosticos.sort_values("MAE").reset_index(drop=True)

print("Pronóstico 2025 de todas las combinaciones (ordenadas de mejor a peor):")
print(pronosticos.round(0).to_string(index=False))

final = pronosticos.iloc[0]
print("\n================ PRONÓSTICO FINAL 2025 ================")
print(f"Modelo:            {final['modelo']} ({int(final['ventana'])} años)")
print(f"Hectáreas 2024:    {serie.loc[2024]:,.0f}")
print(f"Pronóstico 2025:   {final['pronostico_2025']:,.0f} ha "
      f"({final['pronostico_2025'] / serie.loc[2024] - 1:+.1%} vs 2024)")
print(f"Rango orientativo: {final['pronostico_2025'] - final['MAE']:,.0f} – "
      f"{final['pronostico_2025'] + final['MAE']:,.0f} ha  (± el error promedio de la prueba)")

# %%
# ============================================================
# 11. GRÁFICO FINAL
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(serie.index, serie.values, marker="o", color="black", label="Real")
plt.errorbar(ANIO_PRONOSTICO, final["pronostico_2025"], yerr=final["MAE"], fmt="o", color="#2a78d6",
             capsize=6, markersize=9, label=f"Pronóstico 2025 – {final['modelo']}")
plt.plot([2024, 2025], [serie.loc[2024], final["pronostico_2025"]], "--", color="#2a78d6")
plt.title("Hectáreas de coca: histórico y pronóstico 2025")
plt.ylabel("Hectáreas")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
