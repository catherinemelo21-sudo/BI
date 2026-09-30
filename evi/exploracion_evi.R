# =============================================================================
# Proyecto Visualización de Datos - Parte 1
# Encuesta de Visitantes Internacionales (EVI) - DANE
# Periodos: Agosto 2023 - Julio 2024 y Enero - Junio 2025
#
# Contenido (alineado con la rúbrica):
#   0. Carga de datos
#   1. Renombrar variables
#   2. Tipos de variables (requisito: >= 10 numéricas y >= 5 categóricas)
#   3. Diccionario de datos
#   4. Calidad de los datos (faltantes, atípicos, inconsistencias)
#   5. Estadística descriptiva de variables numéricas
#   6. Frecuencias de variables categóricas
#   7. Gráficos exploratorios
#   8. Cifras ponderadas con factor de expansión (para los insights)
# =============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(scales)

dir.create("evi/resultados", showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 0. Carga de datos
# -----------------------------------------------------------------------------
# El DANE publica un archivo por mes/periodo. Pon todos los archivos del
# periodo en evi/datos/ y se unen en una sola tabla.
# Si tus archivos son .csv separados por ";" usa read_csv2 o read_delim.
archivos <- list.files("evi/datos", pattern = "\\.csv$", full.names = TRUE)

datos <- archivos %>%
  lapply(function(f) {
    read_delim(f, delim = ";", locale = locale(decimal_mark = ",", encoding = "latin1"),
               col_types = cols(.default = col_character())) %>%
      mutate(archivo_origen = basename(f))
  }) %>%
  bind_rows()

# Todo se lee como texto para que bind_rows no falle si un mes trae un tipo
# distinto; los tipos correctos se asignan en la sección 2.

# -----------------------------------------------------------------------------
# 1. Renombrar variables
# -----------------------------------------------------------------------------
# Se usa un vector nuevo = "viejo" con any_of(): si alguna pregunta no existe
# en un periodo (el formulario 2025 cambió frente al de 2023-2024), no da error.
nombres <- c(
  modo_transporte = "Modo",
  id_formulario = "DIRECTORIO",
  periodo_encuesta = "Numpub",
  anio_recoleccion = "Vigencia",
  mes_recoleccion = "Mes",
  punto_encuesta = "Aeropuerto_puerto",
  tipo_punto_encuesta = "Tipo",
  unidad_muestreo = "UPM",

  reside_colombia = "P3807",
  pais_residencia = "P3808",
  tipo_visitante = "Visitante",
  esta_en_transito = "P3834",
  salio_puerto_aeropuerto = "P3820",
  pais_destino = "P3822",
  motivo_viaje = "P3821",
  tipo_tratamiento_medico = "P3821S2",
  grupo_viaje_confirmado = "P3809",
  viaja_con_mascotas = "P3837",
  noches_viaje = "P3817S1",
  tamano_grupo_viaje = "P3835",
  personas_grupo_residen_colombia = "P3836",

  gasto_total_grupo_usd = "P3810",
  gasto_paquete_turistico_usd = "P3825S2A4",
  gasto_alojamiento_usd = "P3825S1A4",
  gasto_alimentos_bebidas_usd = "P3825S4A4",
  gasto_transporte_usd = "P3825S9A4",
  gasto_servicios_culturales_usd = "P3825S12A4",
  gasto_servicios_deportivos_usd = "P3825S15A4",
  gasto_artesanias_souvenires_regalos_usd = "P3825S10A4",
  gasto_bienes_uso_personal_usd = "P3825S11A4",
  gasto_joyas_accesorios_lujo_usd = "P3825S16A4",
  gasto_reventa_negocio_usd = "P3825S13A4",
  gasto_mascotas_usd = "P3825S17A4",
  gasto_tratamientos_bienestar_usd = "P3825S18A4",
  gasto_otros_viaje_usd = "P3825S14A4",
  gasto_crucero_usd = "P3825S20A4",

  tour_adquirido_crucero = "P3857S1",
  tour_adquirido_linea = "P3857S2",
  tour_adquirido_puerto_ciudad = "P3857S3",

  actividad_musica_danza_teatro_cine_moda = "P3828S7",
  actividad_comunidades = "P3828S8",
  actividad_recreacion_naturaleza = "P3828S9",
  actividad_tratamientos_alternativos = "P3828S10",
  actividad_recreacion_playas = "P3828S11",
  actividad_bienes_interes_cultural = "P3828S1",
  actividad_fiestas_festividades_carnavales = "P3828S2",
  actividad_artesanias_oficios_tradicionales = "P3828S3",
  actividad_otra = "P3828S5",

  viaje_agencia_tradicional = "P3826A1",
  viaje_agencia_online = "P3826A2",
  viaje_cuenta_propia = "P3826A3",

  paquete_incluye_alojamiento = "P3826S1",
  paquete_incluye_alimentos_bebidas = "P3826S3",
  paquete_incluye_transporte_internacional = "P3826S2",
  paquete_incluye_transporte_interno = "P3826S6",
  paquete_incluye_museos_espectaculos = "P3826S4",
  paquete_incluye_tours = "P3826S5",
  paquete_incluye_alquiler_vehiculo = "P3826S8",
  paquete_incluye_crucero = "P3826S10",
  paquete_incluye_servicios_deportivos = "P3826S9",
  paquete_incluye_otro = "P3826S7",

  reserva_alojamiento_airbnb = "P3844S1",
  reserva_alojamiento_booking = "P3844S2",
  reserva_alojamiento_hoteles_com = "P3844S3",
  reserva_alojamiento_kayak = "P3844S4",
  reserva_alojamiento_trivago = "P3844S5",
  reserva_alojamiento_expedia = "P3844S6",
  reserva_alojamiento_otra_plataforma = "P3844S7",
  reserva_alojamiento_ninguna = "P3844S8",

  destino_principal = "Per_prin",
  noches_destino_principal = "Noches_per_prin",
  continente_destino_principal = "Cont_per_prin",
  medio_conocimiento_colombia = "P3829",
  recomendaria_destino = "P3831",
  motivo_no_recomendacion = "P3840",

  factor_expansion_viajes = "fexp",
  factor_expansion_visitantes = "fexp_personas"
)

# Revisar qué nombres NO se encontraron (typos o preguntas que no existen)
no_encontradas <- setdiff(nombres, names(datos))
if (length(no_encontradas) > 0) {
  message("Columnas no encontradas en los datos: ", paste(no_encontradas, collapse = ", "))
}

datos <- datos %>% rename(any_of(nombres))

# -----------------------------------------------------------------------------
# 2. Tipos de variables
# -----------------------------------------------------------------------------
vars_numericas <- c(
  "noches_viaje", "tamano_grupo_viaje", "personas_grupo_residen_colombia",
  "noches_destino_principal",
  grep("^gasto_", names(datos), value = TRUE),
  "factor_expansion_viajes", "factor_expansion_visitantes"
)
vars_numericas <- intersect(vars_numericas, names(datos))

# Todo lo que no es numérico ni identificador se trata como categórico
vars_id <- c("id_formulario", "unidad_muestreo", "archivo_origen")
vars_categoricas <- setdiff(names(datos), c(vars_numericas, vars_id))

datos <- datos %>%
  mutate(
    across(all_of(vars_numericas), ~ parse_number(.x, locale = locale(decimal_mark = ","))),
    across(all_of(vars_categoricas), as.factor)
  )

# Variables de análisis (sin factores de expansión)
vars_analisis_num <- setdiff(vars_numericas, c("factor_expansion_viajes", "factor_expansion_visitantes"))

cat("Registros:", nrow(datos), "\n")
cat("Variables totales:", ncol(datos), "\n")
cat("Variables numéricas de análisis:", length(vars_analisis_num), "(mínimo 10)\n")
cat("Variables categóricas:", length(vars_categoricas), "(mínimo 5)\n")

# NOTA: las etiquetas de los códigos (1 = Sí, 2 = No, código de país, motivo
# de viaje, etc.) deben tomarse del diccionario de datos del DANE (metadatos
# de la EVI en microdatos.dane.gov.co). Ejemplo para una pregunta Sí/No:
# datos <- datos %>%
#   mutate(reside_colombia = factor(reside_colombia, levels = c(1, 2), labels = c("Sí", "No")))

# -----------------------------------------------------------------------------
# 3. Diccionario de datos (base para completar la descripción en el informe)
# -----------------------------------------------------------------------------
diccionario <- tibble(
  variable = names(datos),
  nombre_original = ifelse(names(datos) %in% names(nombres),
                           unname(nombres[names(datos)]), names(datos)),
  tipo = case_when(
    names(datos) %in% vars_numericas ~ "Numérica",
    names(datos) %in% vars_id ~ "Identificador",
    TRUE ~ "Categórica"
  ),
  n_categorias = sapply(datos, function(x) if (is.factor(x)) nlevels(x) else NA_integer_),
  pct_faltantes = round(colMeans(is.na(datos)) * 100, 1),
  descripcion = ""  # completar a mano con el diccionario del DANE
)
write_csv(diccionario, "evi/resultados/diccionario_datos.csv")

# -----------------------------------------------------------------------------
# 4. Calidad de los datos
# -----------------------------------------------------------------------------
# 4.1 Faltantes
faltantes <- diccionario %>%
  select(variable, tipo, pct_faltantes) %>%
  arrange(desc(pct_faltantes))
print(faltantes, n = 30)
write_csv(faltantes, "evi/resultados/faltantes.csv")

# Ojo: en las preguntas de gasto un NA suele significar "no aplica / no gastó
# en ese rubro" (pregunta filtrada), no un dato perdido. Documéntalo así.

# 4.2 Duplicados
cat("Formularios duplicados:", sum(duplicated(datos$id_formulario)), "\n")

# 4.3 Atípicos con la regla de Tukey (1.5 * IQR)
atipicos <- datos %>%
  select(all_of(vars_analisis_num)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "valor") %>%
  filter(!is.na(valor)) %>%
  group_by(variable) %>%
  summarise(
    q1 = quantile(valor, 0.25),
    q3 = quantile(valor, 0.75),
    limite_superior = q3 + 1.5 * (q3 - q1),
    n_atipicos = sum(valor > limite_superior | valor < q1 - 1.5 * (q3 - q1)),
    pct_atipicos = round(100 * n_atipicos / n(), 1),
    maximo = max(valor),
    .groups = "drop"
  ) %>%
  arrange(desc(pct_atipicos))
print(atipicos, n = 30)
write_csv(atipicos, "evi/resultados/atipicos.csv")

# 4.4 Inconsistencias
inconsistencias <- tibble(
  regla = c(
    "Gasto negativo",
    "Noches en destino principal > noches totales del viaje",
    "Personas que residen en Colombia > tamaño del grupo",
    "Tamaño de grupo <= 0",
    "Suma de rubros de gasto > gasto total del grupo"
  ),
  n_casos = c(
    sum(datos %>% select(starts_with("gasto_")) < 0, na.rm = TRUE),
    sum(datos$noches_destino_principal > datos$noches_viaje, na.rm = TRUE),
    sum(datos$personas_grupo_residen_colombia > datos$tamano_grupo_viaje, na.rm = TRUE),
    sum(datos$tamano_grupo_viaje <= 0, na.rm = TRUE),
    sum(
      rowSums(select(datos, starts_with("gasto_"), -gasto_total_grupo_usd), na.rm = TRUE) >
        datos$gasto_total_grupo_usd + 1,
      na.rm = TRUE
    )
  )
)
print(inconsistencias)
write_csv(inconsistencias, "evi/resultados/inconsistencias.csv")

# -----------------------------------------------------------------------------
# 5. Estadística descriptiva de variables numéricas
# -----------------------------------------------------------------------------
descriptivas <- datos %>%
  select(all_of(vars_analisis_num)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "valor") %>%
  group_by(variable) %>%
  summarise(
    n = sum(!is.na(valor)),
    faltantes = sum(is.na(valor)),
    media = mean(valor, na.rm = TRUE),
    mediana = median(valor, na.rm = TRUE),
    desv_estandar = sd(valor, na.rm = TRUE),
    coef_variacion = desv_estandar / media,
    minimo = min(valor, na.rm = TRUE),
    p25 = quantile(valor, 0.25, na.rm = TRUE),
    p75 = quantile(valor, 0.75, na.rm = TRUE),
    maximo = max(valor, na.rm = TRUE),
    rango = maximo - minimo,
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))
print(descriptivas, n = 30, width = Inf)
write_csv(descriptivas, "evi/resultados/descriptivas_numericas.csv")

# Descriptivas del gasto total por grupos (útil para los insights)
gasto_por_grupo <- function(var) {
  datos %>%
    group_by(.data[[var]]) %>%
    summarise(
      n = n(),
      gasto_medio = mean(gasto_total_grupo_usd, na.rm = TRUE),
      gasto_mediano = median(gasto_total_grupo_usd, na.rm = TRUE),
      noches_medianas = median(noches_viaje, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n))
}
print(gasto_por_grupo("motivo_viaje"))
print(gasto_por_grupo("modo_transporte"))
print(gasto_por_grupo("tipo_visitante"))

# -----------------------------------------------------------------------------
# 6. Frecuencias de variables categóricas
# -----------------------------------------------------------------------------
vars_cat_clave <- intersect(
  c("modo_transporte", "tipo_visitante", "motivo_viaje", "pais_residencia",
    "continente_destino_principal", "destino_principal", "punto_encuesta",
    "medio_conocimiento_colombia", "recomendaria_destino", "viaje_cuenta_propia"),
  names(datos)
)

frecuencias <- lapply(vars_cat_clave, function(v) {
  datos %>%
    count(categoria = .data[[v]], name = "n") %>%
    mutate(variable = v, pct = round(100 * n / sum(n), 1)) %>%
    arrange(desc(n))
}) %>%
  bind_rows() %>%
  select(variable, categoria, n, pct)
write_csv(frecuencias, "evi/resultados/frecuencias_categoricas.csv")

for (v in vars_cat_clave) {
  cat("\n---", v, "---\n")
  print(frecuencias %>% filter(variable == v) %>% head(10))
}

# -----------------------------------------------------------------------------
# 7. Gráficos exploratorios
# -----------------------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))

# 7.1 Distribución del gasto total (escala log por la fuerte asimetría)
g1 <- datos %>%
  filter(gasto_total_grupo_usd > 0) %>%
  ggplot(aes(gasto_total_grupo_usd)) +
  geom_histogram(bins = 50, fill = "#2a6f97") +
  scale_x_log10(labels = dollar) +
  labs(title = "Distribución del gasto total del grupo de viaje",
       x = "Gasto total (USD, escala logarítmica)", y = "Formularios")
ggsave("evi/resultados/01_hist_gasto_total.png", g1, width = 8, height = 5)

# 7.2 Distribución de noches de viaje
g2 <- datos %>%
  filter(!is.na(noches_viaje), noches_viaje <= quantile(noches_viaje, 0.99, na.rm = TRUE)) %>%
  ggplot(aes(noches_viaje)) +
  geom_histogram(binwidth = 1, fill = "#2a6f97") +
  labs(title = "Noches de permanencia en Colombia (hasta el percentil 99)",
       x = "Noches", y = "Formularios")
ggsave("evi/resultados/02_hist_noches.png", g2, width = 8, height = 5)

# 7.3 Gasto total por motivo de viaje (boxplot)
g3 <- datos %>%
  filter(gasto_total_grupo_usd > 0, !is.na(motivo_viaje)) %>%
  ggplot(aes(reorder(motivo_viaje, gasto_total_grupo_usd, median), gasto_total_grupo_usd)) +
  geom_boxplot(fill = "#a9d6e5", outlier.alpha = 0.2) +
  scale_y_log10(labels = dollar) +
  coord_flip() +
  labs(title = "Gasto total del grupo según motivo de viaje",
       x = "Motivo de viaje", y = "Gasto total (USD, escala log)")
ggsave("evi/resultados/03_box_gasto_motivo.png", g3, width = 8, height = 5)

# 7.4 Top 15 países de residencia
g4 <- datos %>%
  count(pais_residencia, sort = TRUE) %>%
  slice_head(n = 15) %>%
  ggplot(aes(reorder(pais_residencia, n), n)) +
  geom_col(fill = "#2a6f97") +
  coord_flip() +
  labs(title = "Top 15 países de residencia de los visitantes (formularios)",
       x = NULL, y = "Formularios")
ggsave("evi/resultados/04_top_paises.png", g4, width = 8, height = 6)

# 7.5 Composición del gasto por rubro (mediana entre quienes gastaron)
g5 <- datos %>%
  select(starts_with("gasto_"), -gasto_total_grupo_usd) %>%
  pivot_longer(everything(), names_to = "rubro", values_to = "valor") %>%
  group_by(rubro) %>%
  summarise(total = sum(valor, na.rm = TRUE), .groups = "drop") %>%
  mutate(rubro = gsub("^gasto_|_usd$", "", rubro),
         pct = total / sum(total)) %>%
  ggplot(aes(reorder(rubro, pct), pct)) +
  geom_col(fill = "#2a6f97") +
  scale_y_continuous(labels = percent) +
  coord_flip() +
  labs(title = "Participación de cada rubro en el gasto total (muestra)",
       x = NULL, y = "% del gasto")
ggsave("evi/resultados/05_composicion_gasto.png", g5, width = 8, height = 6)

# 7.6 Faltantes por variable
g6 <- faltantes %>%
  filter(pct_faltantes > 0) %>%
  slice_head(n = 30) %>%
  ggplot(aes(reorder(variable, pct_faltantes), pct_faltantes)) +
  geom_col(fill = "#c1121f") +
  coord_flip() +
  labs(title = "Porcentaje de valores faltantes (30 variables con más NA)",
       x = NULL, y = "% faltante")
ggsave("evi/resultados/06_faltantes.png", g6, width = 8, height = 8)

# -----------------------------------------------------------------------------
# 8. Cifras expandidas (población) con el factor de expansión
# -----------------------------------------------------------------------------
# Los conteos anteriores son de la MUESTRA. Para hablar de visitantes o viajes
# reales en el informe, pondera con los factores de expansión.
expandido <- datos %>%
  group_by(anio_recoleccion, motivo_viaje) %>%
  summarise(
    viajes_expandidos = sum(factor_expansion_viajes, na.rm = TRUE),
    visitantes_expandidos = sum(factor_expansion_visitantes, na.rm = TRUE),
    gasto_medio_ponderado = weighted.mean(gasto_total_grupo_usd, factor_expansion_viajes, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(anio_recoleccion, desc(visitantes_expandidos))
print(expandido, n = 40)
write_csv(expandido, "evi/resultados/cifras_expandidas_motivo.csv")
