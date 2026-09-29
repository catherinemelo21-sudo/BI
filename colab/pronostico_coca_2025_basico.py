# %% [markdown]
# # Pronóstico de coca 2025 – versión básica (total nacional)
#
# Comparamos 4 modelos para pronosticar las hectáreas de coca de **2025 a nivel nacional**:
# regresión lineal, regresión polinómica, Holt y ARIMA.
#
# **¿Cómo elegimos el mejor?** Hacemos una "prueba en el pasado": cada modelo pronostica los años
# 2019 a 2024 usando sólo los años anteriores, y medimos cuánto se equivocó (MAE y RMSE).
# Probamos cada modelo con ventanas de **3, 5 y 10 años** de historia.
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
# 2. CARGAR LOS DATOS DESDE DRIVE
# ============================================================
from google.colab import drive
drive.mount("/content/drive")

RUTA_COCA = "/content/drive/MyDrive/Proyecto_Policia/Datos/Detección_de_Cultivos_de_Coca_(hectáreas)_20260831.csv"


def parsear_hectareas(valor):
    """Convierte el texto a número. Corrige el año 2020, que viene como '8.832,92'."""
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0
    texto = str(valor).strip()
    if "," in texto:                                  # formato europeo: 8.832,92 -> 8832.92
        texto = texto.replace(".", "").replace(",", ".")
    return float(texto)


df_coca_file = pd.read_csv(RUTA_COCA, dtype=str)
df_coca_file.columns = df_coca_file.columns.str.strip().str.upper()
df_coca_file["cod_municipio"] = df_coca_file["CODMPIO"].str.zfill(5)

# Formato largo: una fila por municipio y año (todos los años disponibles, 2001-2024)
columnas_anio = [c for c in df_coca_file.columns if c.isdigit()]
fact_cultivos_raw = df_coca_file.melt(id_vars="cod_municipio", value_vars=columnas_anio,
                                      var_name="anio", value_name="valor")
fact_cultivos_raw["anio"] = fact_cultivos_raw["anio"].astype(int)
fact_cultivos_raw["hectareas_coca"] = fact_cultivos_raw["valor"].apply(parsear_hectareas)
fact_cultivos_raw = fact_cultivos_raw[["cod_municipio", "anio", "hectareas_coca"]]

# Serie del TOTAL NACIONAL: suma de todos los municipios por año
serie = fact_cultivos_raw.groupby("anio")["hectareas_coca"].sum()
print(serie.round(0))

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
# 4. PARÁMETROS DE LA PRUEBA
# ============================================================
VENTANAS = [3, 5, 10]                    # años de historia que ve cada modelo
ANIOS_PRUEBA = list(range(2019, 2025))   # años que pronosticamos "en el pasado" para medir el error
ANIO_PRONOSTICO = 2025

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
    return Holt(y, initialization_method="estimated").fit().forecast(1)[0]


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
tabla_mae.plot(kind="bar", figsize=(9, 4), color=["#86b6ef", "#2a78d6", "#104281"], rot=0)
plt.title("Error promedio (MAE) al pronosticar 2019-2024 – menor es mejor")
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
plt.plot(serie.loc[2012:].index, serie.loc[2012:].values, marker="o", color="black", label="Real")
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
