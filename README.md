# Pronóstico de cultivos de coca 2025

Notebook que compara **regresión lineal, regresión polinómica, Holt y ARIMA** (con ventanas de 3, 5 y 10 años)
para pronosticar las hectáreas de coca de 2025, por municipio y para el total nacional.

| Archivo | Contenido |
|---|---|
| `pronostico_coca_2025.ipynb` | Notebook principal, ya ejecutado y explicado paso a paso (abrir en Colab o Jupyter) |
| `reportes/pronostico_coca_2025.html` | El mismo notebook en HTML, para leerlo sin ejecutar nada |
| `resultados/pronosticos_coca_2025.xlsx` | Tablas finales: total nacional, municipios, departamentos y errores de los modelos |
| `resultados/*.csv`, `resultados/*.png` | Las mismas tablas en CSV y los gráficos |
| `data/` | CSV fuente: *Detección de Cultivos de Coca (hectáreas)*, SIMCI / Datos Abiertos, 2001-2024 |
| `scripts/construir_notebook.py` | Genera el notebook (para editarlo y revisarlo en git) |

## Resultados principales

| | 2024 (real) | **2025 (pronóstico)** | Rango orientativo | Modelo |
|---|---|---|---|---|
| Total nacional | 261.386 ha | **270.138 ha (+3,3%)** | 252.654 – 287.622 | Regresión lineal, 10 años |

- **Nacional:** la regresión lineal con 10 años tuvo el menor error al pronosticar 2019-2024 (MAE ≈ 17.500 ha, ~8%),
  y le gana a la referencia "igual al año anterior" (≈ 24.100 ha). Las ventanas cortas persiguen los giros de un solo año.
- **Municipios:** ARIMA con 10 años fue el mejor de los 4 modelos, pero no superó a la referencia ingenua; úsense
  los valores municipales como ranking/orden de magnitud. Top 5 en 2025: Tumaco, Tibú, El Tambo, El Charco, Puerto Asís.

## ⚠️ Corrección de datos (afecta al notebook de limpieza)

La columna **2020** del CSV usa formato europeo (`8.832,92`). La función `parsear_hectareas` del notebook de
limpieza la convierte en `8.832.92`, falla y devuelve **0**: se pierden ~115.500 ha (81% del total 2020) en
`fact_cultivos_raw` / `FactCultivos`. Este notebook usa una versión corregida:

```python
def parsear_hectareas(valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return 0.0
    texto = str(valor).strip()
    if "," in texto:                       # formato europeo: 8.832,92
        texto = texto.replace(".", "").replace(",", ".")
    return float(texto)
```

## Ejecutar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace pronostico_coca_2025.ipynb
```
En Colab, cambiar `RUTA_COCA` en la celda de configuración por la ruta del CSV en Drive.
