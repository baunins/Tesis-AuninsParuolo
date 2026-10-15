# Exposición ocupacional a inteligencia artificial y reacción de retornos ante shocks macroeconómicos

Código de replicación de la tesis de grado de Benjamín Aunins y Francesca Paruolo
(Licenciatura en Finanzas, Universidad de San Andrés, 2026).

## Scripts

| Script | Qué estima |
|---|---|
| `regresiones_paula.py` | Especificación base, contraste sobre la dispersión y respuesta acumulada (proyecciones locales). |
| `regresiones_periodos.py` | Efecto por período (hasta 2011, 2012–2022 y desde el lanzamiento de ChatGPT), en la muestra completa y en las firmas que permanecieron en el índice. Apéndice B. |
| `regresiones_simultaneo.py` | Correlaciones entre los shocks y estimación con los tres shocks en simultáneo. Sección 3.3. |

## Datos

Los datos de mercado y de balance provienen de Sharadar (Core US Equities), adquiridos
con una licencia de uso personal que **no permite redistribuirlos**. Por ese motivo el
repositorio no incluye el panel (`panel_eis.csv`) ni ningún archivo de datos.

Data from Sharadar.

## Uso

```
pip install -r requirements.txt
python regresiones_paula.py
python regresiones_periodos.py
python regresiones_simultaneo.py
```

Cada script lee `panel_eis.csv` desde la carpeta de trabajo y escribe sus resultados en
una carpeta propia.
